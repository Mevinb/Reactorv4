"""
ReactorV4 — Main Pipeline Orchestrator

Full pipeline:
  Source image ──► Face detect
  Target image ──► inswapper (adaptive, any face) ──► Occlusion preserve
                                  │
                         SafeFaceRestorer
                         RestoreFormer++ / CodeFormer / GFPGAN / GPEN
                         (ONNX, auto-download, no pickle)
                                  │
                    Reference Detail Enhancement (adaptive)
                    (eye + teeth detail injection from source)
                                  │
                        E4S Texture Transfer
                        (reference-guided skin matching)
                                  │
                             Final output
"""

from __future__ import annotations

import ctypes
import gc
import os
import platform
import sys
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

TAG = "[ReactorV4 Pipeline]"

_ext_scripts = os.path.dirname(os.path.abspath(__file__))
if _ext_scripts not in sys.path:
    sys.path.insert(0, _ext_scripts)

from reactor_v4_swapper       import AdaptiveFaceSwapper, get_swapper, _build_feathered_face_mask
from reactor_v4_osdface       import (
    SafeFaceRestorer, get_safe_restorer, clear_restorer_cache,
    DEFAULT_MODEL, RESTORER_REGISTRY, list_restorer_choices
)
from reactor_v4_texture_transfer import E4SStyleTextureTransfer, get_texture_transfer

# Optional WebUI Forge memory management & interrupt support
try:
    _webui_root = os.path.abspath(os.path.join(_ext_scripts, "..", "..", ".."))
    if _webui_root not in sys.path:
        sys.path.insert(0, _webui_root)
    from backend import memory_management as _mm
    _MM = True
except ImportError:
    _mm = None
    _MM = False

try:
    from modules import shared as _shared
    _HAS_SHARED = True
except ImportError:
    _shared = None
    _HAS_SHARED = False


def is_interrupted() -> bool:
    """Check if the user pressed Interrupt/Skip in the Forge UI."""
    if not _HAS_SHARED:
        return False
    st = getattr(_shared, "state", None)
    if st is None:
        return False
    return getattr(st, "interrupted", False) or getattr(st, "skipped", False)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ── Adaptive Reference Detail Enhancement ─────────────────────────────────────
_DETAIL_SIZE = 512  # canonical aligned face size (512 % 128 == 0)


def _transform_kps(kps: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Transform 5-point landmarks through an affine matrix."""
    out = np.zeros((5, 2), dtype=np.float32)
    for i in range(5):
        pt = np.array([kps[i][0], kps[i][1], 1.0], dtype=np.float32)
        out[i] = (M @ pt)[:2]
    return out


def _adaptive_region_strength(
    result_region: np.ndarray,
    source_region: np.ndarray,
) -> float:
    """
    Compute adaptive transfer strength by comparing sharpness (Laplacian
    variance) between the swapped result and the source reference.
    Higher degradation → higher strength.
    """
    if result_region.size < 100 or source_region.size < 100:
        return 0.0
    res_gray = cv2.cvtColor(result_region, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.cvtColor(source_region, cv2.COLOR_BGR2GRAY)
    res_lap = cv2.Laplacian(res_gray, cv2.CV_64F).var()
    src_lap = cv2.Laplacian(src_gray, cv2.CV_64F).var()
    if src_lap < 1.0:
        return 0.15  # source featureless (closed eyes/mouth) — light touch
    ratio = res_lap / max(src_lap, 1e-6)
    # ratio < 0.5 = result lost lots of detail → high strength
    # ratio > 1.0 = result already sharp → minimal
    strength = float(np.clip(1.0 - ratio * 0.7, 0.10, 0.80))
    return strength


def _reference_detail_enhance(
    result_img: np.ndarray,
    source_img: np.ndarray,
    target_img: np.ndarray,
    swapper: "AdaptiveFaceSwapper",
    result_face_idx: int = 0,
    source_face_idx: int = 0,
) -> np.ndarray:
    """
    Target-guided expression and detail preservation for eyes and teeth.

    Aligns both result and original target faces to canonical 512px space,
    and blends the target's eyes/teeth back onto the swapped face.
    This restores the original expression (smile, teeth, gaze direction)
    and prevents low-res swap artifacts from degrading eyes and teeth.
    """
    try:
        from insightface.utils.face_align import estimate_norm
    except ImportError:
        print(f"{TAG} insightface.utils.face_align unavailable — skipping detail enhance")
        return result_img

    # 1. Detect faces on original target image (stable landmarks)
    tgt_faces = swapper._detect_faces_adaptive(target_img)
    if not tgt_faces:
        print(f"{TAG} Detail enhance: no target face detected — skipping")
        return result_img

    tgt_face = tgt_faces[min(result_face_idx, len(tgt_faces) - 1)]
    tgt_kps = getattr(tgt_face, "kps", None)
    if tgt_kps is None:
        return result_img

    tgt_kps = np.array(tgt_kps, dtype=np.float32)
    if tgt_kps.shape != (5, 2):
        return result_img

    # 2. Align result_img and target_img to canonical 512px space using target landmarks
    S = _DETAIL_SIZE
    res_M = estimate_norm(tgt_kps, S)
    res_aligned = cv2.warpAffine(result_img, res_M, (S, S), borderValue=0.0)
    tgt_aligned = cv2.warpAffine(target_img, res_M, (S, S), borderValue=0.0)

    # 3. Get landmark positions in aligned space
    akps = _transform_kps(tgt_kps, res_M)

    # 4. Build eye and mouth region masks
    # Landmarks: 0=left_eye, 1=right_eye, 2=nose, 3=left_mouth, 4=right_mouth
    ew = int(S * 0.08)   # eye ellipse width radius
    eh = int(S * 0.055)  # eye ellipse height radius
    eye_mask = np.zeros((S, S), dtype=np.float32)
    for idx in [0, 1]:
        cx, cy = int(akps[idx][0]), int(akps[idx][1])
        cv2.ellipse(eye_mask, (cx, cy), (ew, eh), 0, 0, 360, 1.0, -1)

    mouth_mask = np.zeros((S, S), dtype=np.float32)
    mcx = int((akps[3][0] + akps[4][0]) / 2)
    mcy = int((akps[3][1] + akps[4][1]) / 2)
    mw = max(10, int(abs(akps[4][0] - akps[3][0]) * 0.75))
    mh = max(8, int(mw * 0.55))
    cv2.ellipse(mouth_mask, (mcx, mcy), (mw, mh), 0, 0, 360, 1.0, -1)

    # Feather masks
    fk = max(5, int(S * 0.035))
    fk += 1 - fk % 2  # ensure odd
    eye_mask = cv2.GaussianBlur(eye_mask, (fk, fk), fk * 0.4)
    mouth_mask = cv2.GaussianBlur(mouth_mask, (fk, fk), fk * 0.4)

    # 5. Blend the original target's eyes and mouth back to the swapped face
    # We use a blend weight of 0.80 for eyes and 0.85 for mouth
    eye_blend_w = 0.80
    mouth_blend_w = 0.85

    enhanced = res_aligned.astype(np.float32)
    tgt_f = tgt_aligned.astype(np.float32)

    em = (eye_mask * eye_blend_w)[:, :, None]
    mm = (mouth_mask * mouth_blend_w)[:, :, None]

    enhanced = enhanced * (1.0 - em) + tgt_f * em
    enhanced = enhanced * (1.0 - mm) + tgt_f * mm

    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    # 6. Warp back to original image space
    combined_mask = np.maximum(eye_mask * eye_blend_w, mouth_mask * mouth_blend_w)
    res_IM = cv2.invertAffineTransform(res_M)
    enhanced_full = cv2.warpAffine(enhanced, res_IM, (result_img.shape[1], result_img.shape[0]), borderValue=0.0)
    full_mask = cv2.warpAffine(combined_mask, res_IM, (result_img.shape[1], result_img.shape[0]), borderValue=0.0)
    full_mask = np.clip(full_mask, 0, 1)[:, :, None]

    output = (
        enhanced_full.astype(np.float32) * full_mask +
        result_img.astype(np.float32) * (1.0 - full_mask)
    ).astype(np.uint8)

    print(f"{TAG} Target expression and detail preservation applied (eyes={eye_blend_w:.2f}, mouth={mouth_blend_w:.2f})")
    return output


# ── Non-swapped face protection ───────────────────────────────────────────────
def _protect_non_swapped_faces(
    original: np.ndarray,
    processed: np.ndarray,
    swapper: "AdaptiveFaceSwapper",
    swapped_indices: list,
) -> np.ndarray:
    """
    Restore non-swapped faces from the original image.
    Only the face(s) that were actually swapped keep the pipeline processing.
    All other faces get their original pixels blended back.
    """
    faces = swapper._detect_faces_adaptive(original)
    if not faces or len(faces) <= 1:
        return processed  # single face or none — nothing to protect

    result = processed.copy()
    protected = 0
    for i, face in enumerate(faces):
        if i in swapped_indices:
            continue  # swapped face — keep processed version
        # Build a feathered mask for this non-swapped face and restore original pixels
        mask, _ = _build_feathered_face_mask(face, original.shape, expand_ratio=0.15)
        mask_3ch = mask[:, :, None]
        result = (
            original.astype(np.float32) * mask_3ch +
            result.astype(np.float32) * (1.0 - mask_3ch)
        ).astype(np.uint8)
        protected += 1

    if protected > 0:
        print(f"{TAG} Protected {protected} non-swapped face(s) from pipeline modifications")
    return result


class ReactorV4Pipeline:
    """
    ReactorV4 main pipeline.
    All three steps are individually togglable.
    Restorer model is selectable at runtime via the UI.
    """

    def __init__(self, models_path: str):
        self.models_path = models_path
        self._lock       = threading.RLock()
        self._is_linux   = platform.system().lower() == "linux"
        self._device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Sub-modules (lazy)
        self._swapper:  Optional[AdaptiveFaceSwapper]      = None
        self._restorer: Optional[SafeFaceRestorer]          = None
        self._texture:  Optional[E4SStyleTextureTransfer]   = None

        # Settings
        self.restorer_model:       str   = DEFAULT_MODEL
        self.restorer_strength:    float = 1.0
        self.enable_restoration:   bool  = True
        self.enable_texture:       bool  = True
        self.texture_strength:     float = _env_float("REACTOR_V4_TEXTURE_STRENGTH", 0.45)
        self.hf_strength:          float = _env_float("REACTOR_V4_HF_STRENGTH",      0.35)
        self.swap_strength:        float = 1.0
        self.gender_match:         str   = "S"
        self.occlusion_enabled:    bool  = True
        self.occlusion_strength:   float = 1.0
        self.occlusion_sensitivity: float = 0.55
        self.enable_detail_enhance: bool = True
        self.auto_cleanup:         bool  = True
        self.aggressive_cleanup:   bool  = False

        print(f"{TAG} Device: {self._device}")
        print(f"{TAG} Default restorer: {self.restorer_model}")

    # ── Lazy sub-modules ──────────────────────────────────────────────────────
    def _get_swapper(self) -> AdaptiveFaceSwapper:
        if self._swapper is None:
            self._swapper = get_swapper(self.models_path)
        self._swapper.occlusion_enabled     = self.occlusion_enabled
        self._swapper.occlusion_strength    = self.occlusion_strength
        self._swapper.occlusion_sensitivity = self.occlusion_sensitivity
        return self._swapper

    def _get_restorer(self) -> SafeFaceRestorer:
        if self._restorer is None:
            self._restorer = get_safe_restorer(self.models_path, self.restorer_model)
        else:
            self._restorer.set_model(self.restorer_model)
        return self._restorer

    def _get_texture(self) -> E4SStyleTextureTransfer:
        if self._texture is None:
            self._texture = get_texture_transfer(self._device)
        return self._texture

    # ── Face bbox helper ──────────────────────────────────────────────────────
    def _face_bbox(self, swapper: AdaptiveFaceSwapper, img: np.ndarray, idx: int):
        try:
            faces = swapper._detect_faces_adaptive(img)
            if not faces:
                return None
            f = faces[min(idx, len(faces) - 1)]
            h, w = img.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            return max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        except Exception:
            return None

    # ── Single-face pipeline ──────────────────────────────────────────────────
    def process(
        self,
        source_img: np.ndarray,
        target_img: np.ndarray,
        source_face_idx: int = 0,
        target_face_idx: int = 0,
        swapper_model: str = "inswapper_128.onnx",
        source_img_2: Optional[np.ndarray] = None,
        source_face_idx_2: int = 0,
        target_face_idx_2: int = 0,
    ) -> Tuple[np.ndarray, str]:
        t0 = time.time()
        print(f"\n{TAG} ══ ReactorV4 Pipeline ══")
        print(f"{TAG} Restorer: {self.restorer_model} | Texture: {self.enable_texture}")

        with self._lock:
            try:
                # ── Interrupt gate ─────────────────────────────────────
                if is_interrupted():
                    print(f"{TAG} ✋ Interrupted before start")
                    return target_img.copy(), "Interrupted"

                # Auto-matching genders for dual swap (couple support)
                gender_match_1 = self.gender_match
                gender_match_2 = self.gender_match
                
                if source_img_2 is not None:
                    try:
                        swapper = self._get_swapper()
                        src_faces_1 = swapper._detect_faces_adaptive(source_img)
                        src_faces_2 = swapper._detect_faces_adaptive(source_img_2)
                        tgt_faces = swapper._detect_faces_adaptive(target_img)
                        
                        if src_faces_1 and src_faces_2 and len(tgt_faces) >= 2:
                            f1 = src_faces_1[min(source_face_idx, len(src_faces_1) - 1)]
                            f2 = src_faces_2[min(source_face_idx_2, len(src_faces_2) - 1)]
                            g1 = swapper._gender(f1)
                            g2 = swapper._gender(f2)
                            
                            if g1 != g2 and g1 in ("M", "F") and g2 in ("M", "F"):
                                t1_idx = None
                                for idx, tf in enumerate(tgt_faces):
                                    if swapper._gender(tf) == g1:
                                        t1_idx = idx
                                        break
                                
                                t2_idx = None
                                for idx, tf in enumerate(tgt_faces):
                                    if swapper._gender(tf) == g2 and idx != t1_idx:
                                        t2_idx = idx
                                        break
                                
                                if t1_idx is not None and t2_idx is not None:
                                    print(f"{TAG} Couple auto-match: Face 1 ({g1}) -> Target {t1_idx}, Face 2 ({g2}) -> Target {t2_idx}")
                                    target_face_idx = t1_idx
                                    target_face_idx_2 = t2_idx
                                    gender_match_1 = "A"
                                    gender_match_2 = "A"
                    except Exception as e:
                        print(f"{TAG} Couple auto-match helper error: {e}")

                # Step 1 — Swap
                print(f"\n{TAG} Step 1/3 — Adaptive Face Swap (Face 1)")
                swapper = self._get_swapper()
                result, swap_msg, actual_tgt_idx = swapper.swap_faces(
                    source_img, target_img,
                    source_face_idx, target_face_idx,
                    swapper_model, gender_match_1, self.swap_strength,
                )
                if "Error" in swap_msg:
                    return target_img.copy(), swap_msg

                # Option: Swap Face 2
                actual_tgt_idx_2 = None
                swap_2_applied = False
                if source_img_2 is not None:
                    # Detect target faces on original target image
                    tgt_faces = swapper._detect_faces_adaptive(target_img)
                    if len(tgt_faces) < 2:
                        print(f"{TAG} ℹ Target image only has {len(tgt_faces)} face(s) — skipping second face swap to prevent duplicate swap.")
                    else:
                        print(f"\n{TAG} Step 1b — Adaptive Face Swap (Face 2)")
                        result, swap_msg_2, actual_tgt_idx_2 = swapper.swap_faces(
                            source_img_2, result,
                            source_face_idx_2, target_face_idx_2,
                            swapper_model, gender_match_2, self.swap_strength,
                        )
                        if "Error" in swap_msg_2:
                            print(f"{TAG} ⚠ Second swap failed: {swap_msg_2}")
                        else:
                            swap_msg += f" | {swap_msg_2}"
                            swap_2_applied = True

                if is_interrupted():
                    print(f"{TAG} ✋ Interrupted after swap")
                    return target_img.copy(), "Interrupted after swap"

                # Step 2 — Restore
                if self.enable_restoration and self.restorer_model != "none":
                    print(f"\n{TAG} Step 2/3 — Face Restoration [{self.restorer_model}]")
                    restorer = self._get_restorer()
                    result = restorer.restore(result, strength=self.restorer_strength)
                else:
                    print(f"{TAG} Step 2/3 — Restoration SKIPPED")

                if is_interrupted():
                    print(f"{TAG} ✋ Interrupted after restore")
                    return target_img.copy(), "Interrupted after restore"

                # Step 2.5 — Reference Detail Enhancement (eyes + teeth)
                if self.enable_detail_enhance:
                    print(f"\n{TAG} Step 2.5 — Reference Detail Enhancement (adaptive)")
                    result = _reference_detail_enhance(
                        result, source_img, target_img, swapper,
                        result_face_idx=target_face_idx,
                        source_face_idx=source_face_idx,
                    )
                    if source_img_2 is not None and swap_2_applied:
                        print(f"{TAG} Step 2.5b — Detail Enhancement (Face 2)")
                        result = _reference_detail_enhance(
                            result, source_img_2, target_img, swapper,
                            result_face_idx=target_face_idx_2,
                            source_face_idx=source_face_idx_2,
                        )
                else:
                    print(f"{TAG} Step 2.5 — Detail Enhancement SKIPPED")

                if is_interrupted():
                    print(f"{TAG} ✋ Interrupted after detail enhance")
                    return target_img.copy(), "Interrupted after detail enhance"

                # Step 3 — Texture Transfer
                if self.enable_texture and self.texture_strength > 0.0:
                    print(f"\n{TAG} Step 3/3 — E4S Texture Transfer")
                    texture = self._get_texture()
                    # First face
                    bbox = self._face_bbox(swapper, result, target_face_idx)
                    result = texture.transfer(
                        result, source_img,
                        face_bbox=bbox,
                        strength=self.texture_strength,
                        hf_strength=self.hf_strength,
                        protect_non_skin=True,
                    )
                    # Second face
                    if source_img_2 is not None and swap_2_applied:
                        print(f"{TAG} Step 3b — E4S Texture Transfer (Face 2)")
                        bbox_2 = self._face_bbox(swapper, result, target_face_idx_2)
                        result = texture.transfer(
                            result, source_img_2,
                            face_bbox=bbox_2,
                            strength=self.texture_strength,
                            hf_strength=self.hf_strength,
                            protect_non_skin=True,
                        )
                else:
                    print(f"{TAG} Step 3/3 — Texture Transfer SKIPPED")

                # Step 4 — Protect non-swapped faces
                swapped_indices = [actual_tgt_idx]
                if source_img_2 is not None and swap_2_applied and actual_tgt_idx_2 is not None:
                    swapped_indices.append(actual_tgt_idx_2)
                result = _protect_non_swapped_faces(
                    target_img, result, swapper, swapped_indices
                )

                msg = f"Done in {time.time()-t0:.2f}s | {swap_msg}"
                print(f"\n{TAG} ✓ {msg}")
                return result, msg

            except Exception as exc:
                import traceback; traceback.print_exc()
                return target_img.copy(), f"Error: {exc}"
            finally:
                if self.auto_cleanup:
                    self._cleanup(self.aggressive_cleanup)

    # ── Auto all-faces pipeline ───────────────────────────────────────────────
    def process_auto(
        self,
        source_imgs: List[np.ndarray],
        target_img: np.ndarray,
        swapper_model: str = "inswapper_128.onnx",
        similarity_threshold: float = 0.30,
    ) -> Tuple[np.ndarray, str]:
        t0 = time.time()
        print(f"\n{TAG} ══ ReactorV4 Auto Pipeline (all faces) ══")

        with self._lock:
            try:
                # Step 1
                print(f"\n{TAG} Step 1/3 — Auto-match Swap")
                swapper = self._get_swapper()
                result, swap_msg = swapper.swap_all_faces(
                    source_imgs, target_img, swapper_model,
                    self.gender_match, self.swap_strength, similarity_threshold,
                )
                if "Error" in swap_msg:
                    return target_img.copy(), swap_msg

                # Step 2
                if self.enable_restoration and self.restorer_model != "none":
                    print(f"\n{TAG} Step 2/3 — Restoration [{self.restorer_model}]")
                    result = self._get_restorer().restore(result, self.restorer_strength)
                else:
                    print(f"{TAG} Step 2/3 — Restoration SKIPPED")

                # Step 3
                if self.enable_texture and self.texture_strength > 0.0:
                    print(f"\n{TAG} Step 3/3 — Texture Transfer")
                    result = self._get_texture().transfer(
                        result, source_imgs[0],
                        face_bbox=None,
                        strength=self.texture_strength * 0.8,
                        hf_strength=self.hf_strength,
                        protect_non_skin=True,
                    )

                msg = f"Auto done in {time.time()-t0:.2f}s | {swap_msg}"
                print(f"\n{TAG} ✓ {msg}")
                return result, msg

            except Exception as exc:
                import traceback; traceback.print_exc()
                return target_img.copy(), f"Error: {exc}"
            finally:
                if self.auto_cleanup:
                    self._cleanup(self.aggressive_cleanup)

    # ── Configure ─────────────────────────────────────────────────────────────
    def configure(
        self,
        restorer_model:        str   = DEFAULT_MODEL,
        restorer_strength:     float = 1.0,
        enable_restoration:    bool  = True,
        enable_texture:        bool  = True,
        texture_strength:      float = 0.45,
        hf_strength:           float = 0.35,
        swap_strength:         float = 1.0,
        gender_match:          str   = "S",
        occlusion_enabled:     bool  = True,
        occlusion_strength:    float = 1.0,
        enable_detail_enhance: bool  = True,
        auto_cleanup:          bool  = True,
        aggressive_cleanup:    bool  = False,
    ):
        self.restorer_model       = restorer_model
        self.restorer_strength    = float(restorer_strength)
        self.enable_restoration   = bool(enable_restoration)
        self.enable_texture       = bool(enable_texture)
        self.texture_strength     = float(texture_strength)
        self.hf_strength          = float(hf_strength)
        self.swap_strength        = float(swap_strength)
        self.gender_match         = gender_match
        self.occlusion_enabled    = bool(occlusion_enabled)
        self.occlusion_strength   = float(occlusion_strength)
        self.enable_detail_enhance = bool(enable_detail_enhance)
        self.auto_cleanup         = bool(auto_cleanup)
        self.aggressive_cleanup   = bool(aggressive_cleanup)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def _cleanup(self, aggressive: bool = False):
        gc.collect(); gc.collect()
        if aggressive:
            if self._restorer: self._restorer.release()
            if self._swapper:  self._swapper.release()
            self._restorer = None
            self._swapper  = None
            clear_restorer_cache()
        if _MM and _mm:
            try: _mm.soft_empty_cache(force=aggressive)
            except Exception: pass
        elif torch.cuda.is_available():
            try: torch.cuda.empty_cache()
            except Exception: pass
        if self._is_linux:
            try: ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception: pass
        gc.collect()

    def cleanup_memory(self, aggressive: bool = False):
        with self._lock:
            self._cleanup(aggressive)


# ── Singleton ─────────────────────────────────────────────────────────────────
_pipeline: Optional[ReactorV4Pipeline] = None
_plock = threading.Lock()


def get_reactor_v4_pipeline(models_path: str) -> ReactorV4Pipeline:
    global _pipeline
    with _plock:
        if _pipeline is None:
            _pipeline = ReactorV4Pipeline(models_path)
        return _pipeline
