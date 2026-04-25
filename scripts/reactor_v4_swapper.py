"""
ReactorV4 — Adaptive Face Swapper

Uses InsightFace inswapper for face detection & swap.
Key features vs V3:
  - Adaptive multi-face detection (works on ANY face size/angle)
  - Dynamic ONNX provider selection (CUDA/CPU auto-fallback)
  - Feathered soft-mask paste-back with occlusion awareness
  - Minimal naturalization — leaves the image clean for OSDFace + E4S-texture
"""

from __future__ import annotations

import cv2
import gc
import numpy as np
import os
import platform
import sys
import threading
import time
from typing import List, Optional, Tuple

import torch

# ── Constants ────────────────────────────────────────────────────────────────
TAG = "[ReactorV4 Swapper]"


# ── Interrupt helper (lazy import to avoid circular deps) ────────────────────
def _is_interrupted() -> bool:
    try:
        from reactor_v4_pipeline import is_interrupted
        return is_interrupted()
    except ImportError:
        return False


# ── CUDA path helper (same as V3) ────────────────────────────────────────────
def _setup_cudnn_path() -> bool:
    try:
        import glob
        import site
        added = []
        def _add(var, path):
            if not os.path.exists(path):
                return
            cur = os.environ.get(var, "")
            parts = [p for p in cur.split(os.pathsep) if p]
            if path in parts:
                return
            os.environ[var] = path + os.pathsep + cur if cur else path
            added.append(path)
        for sp in site.getsitepackages():
            nr = os.path.join(sp, "nvidia")
            if not os.path.isdir(nr):
                continue
            for pkg in glob.glob(os.path.join(nr, "*")):
                _add("LD_LIBRARY_PATH", os.path.join(pkg, "lib"))
                _add("LD_LIBRARY_PATH", os.path.join(pkg, "lib64"))
                _add("PATH", os.path.join(pkg, "bin"))
        return bool(added)
    except Exception:
        return False


_setup_cudnn_path()

# ── InsightFace lazy import ───────────────────────────────────────────────────
try:
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import model_zoo as mz
    INSIGHTFACE_OK = True
except ImportError:
    INSIGHTFACE_OK = False
    print(f"{TAG} InsightFace not installed — swap unavailable")


# ── Helper: environment flag ──────────────────────────────────────────────────
def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ── Adaptive provider selector ────────────────────────────────────────────────
def _pick_ort_providers(is_linux: bool) -> List[str]:
    """
    Select ONNX Runtime providers. CPU-only on Linux by default for stability.
    Set REACTOR_V4_ENABLE_CUDA=1 to force CUDA on Linux.
    """
    if _env_flag("REACTOR_V4_FORCE_CPU"):
        return ["CPUExecutionProvider"]
    if is_linux and not _env_flag("REACTOR_V4_ENABLE_CUDA") and not _env_flag("REACTOR_V4_FORCE_CUDA"):
        return ["CPUExecutionProvider"]
    if torch.cuda.is_available():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


# ── Geometry helpers ──────────────────────────────────────────────────────────
def _safe_bbox(face, img_h: int, img_w: int) -> Tuple[int, int, int, int]:
    """Clamp bounding box to image bounds."""
    if not hasattr(face, "bbox") or face.bbox is None:
        return 0, 0, img_w, img_h
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(x1 + 1, min(x2, img_w))
    y2 = max(y1 + 1, min(y2, img_h))
    return x1, y1, x2, y2


def _build_feathered_face_mask(
    face,
    img_shape: Tuple[int, int, int],
    expand_ratio: float = 0.12,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Build a soft elliptical mask around the face bounding box.
    Returns (mask float32 [H,W], bbox (x1,y1,x2,y2)).
    """
    h, w = img_shape[:2]
    x1, y1, x2, y2 = _safe_bbox(face, h, w)
    fw, fh = x2 - x1, y2 - y1

    # Expand
    px = max(3, int(fw * expand_ratio))
    py = max(3, int(fh * expand_ratio))
    ex1 = max(0, x1 - px)
    ey1 = max(0, y1 - py)
    ex2 = min(w, x2 + px)
    ey2 = min(h, y2 + py)

    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = (ex1 + ex2) // 2, (ey1 + ey2) // 2
    ax = max(2, (ex2 - ex1) // 2)
    ay = max(2, (ey2 - ey1) // 2)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)

    # Landmark hull boost
    if hasattr(face, "kps") and face.kps is not None:
        kps = np.asarray(face.kps, dtype=np.float32)
        if kps.ndim == 2 and kps.shape[0] >= 3:
            lm = np.zeros_like(mask)
            hull = cv2.convexHull(kps.astype(np.int32))
            cv2.fillConvexPoly(lm, hull, 1.0)
            dk = max(3, int(min(fw, fh) * 0.06))
            kernel = np.ones((dk, dk), dtype=np.uint8)
            lm = cv2.dilate(lm, kernel)
            mask = np.maximum(mask, lm * 0.85)

    feather = max(5, int(min(fw, fh) * 0.12))
    feather += feather % 2 == 0  # must be odd for GaussianBlur
    mask = cv2.GaussianBlur(mask, (feather, feather), feather * 0.35)
    mask = np.clip(mask, 0.0, 1.0)

    return mask.astype(np.float32), (x1, y1, x2, y2)


# ── Occlusion detection ───────────────────────────────────────────────────────
def _detect_occlusions(
    original: np.ndarray,
    swapped: np.ndarray,
    face_mask: np.ndarray,
    sensitivity: float = 0.55,
    strength: float = 1.0,
) -> np.ndarray:
    """
    Return a float32 mask [H,W] of pixels that should be restored from original
    (hair, glasses, hands, etc. that appeared in front of the face).
    Returns zeros array if occlusion system should be skipped.
    """
    if strength <= 0.0:
        return np.zeros(original.shape[:2], dtype=np.float32)

    h, w = original.shape[:2]
    roi = face_mask > 0.10
    if not np.any(roi):
        return np.zeros((h, w), dtype=np.float32)

    gray_o = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_s = cv2.cvtColor(swapped, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_o, gray_s).astype(np.float32)

    roi_vals = diff[roi]
    if roi_vals.size < 20:
        return np.zeros((h, w), dtype=np.float32)

    pct = np.clip(80.0 - sensitivity * 35.0, 35.0, 90.0)
    thr = max(float(np.percentile(roi_vals, pct)), 10.0 + (1.0 - sensitivity) * 20.0)
    strong = diff >= thr

    clow = int(35 + (1.0 - sensitivity) * 45)
    chigh = int(90 + (1.0 - sensitivity) * 95)
    edges_o = cv2.Canny(gray_o, clow, chigh)
    edges_s = cv2.Canny(gray_s, clow, chigh)
    lost = (edges_o > 0) & (edges_s == 0)

    hsv_o = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    hsv_s = cv2.cvtColor(swapped, cv2.COLOR_BGR2HSV)
    sd = cv2.absdiff(hsv_o[:, :, 1], hsv_s[:, :, 1]).astype(np.float32)
    vd = cv2.absdiff(hsv_o[:, :, 2], hsv_s[:, :, 2]).astype(np.float32)
    chroma = (sd >= 12.0 + (1.0 - sensitivity) * 12.0) & (vd >= 20.0)

    face_bin = (face_mask > 0.10).astype(np.uint8)
    bk = max(3, int(min(h, w) * 0.01))
    bkernel = np.ones((bk, bk), dtype=np.uint8)
    inner = cv2.erode(face_bin, bkernel)
    boundary = np.clip(face_bin - inner, 0, 1).astype(np.uint8)

    candidate = (strong & (lost | chroma) & roi).astype(np.uint8)
    if candidate.sum() == 0:
        return np.zeros((h, w), dtype=np.float32)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    filtered = np.zeros_like(candidate)
    min_area = max(14, int(face_bin.sum() * 0.002))
    for lid in range(1, num):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == lid
        if np.any(boundary[comp] > 0):
            filtered[comp] = 1

    if filtered.sum() == 0 or filtered.sum() / max(1, face_bin.sum()) > 0.40:
        return np.zeros((h, w), dtype=np.float32)

    mk = max(3, int(min(h, w) * 0.008))
    mkernel = np.ones((mk, mk), dtype=np.uint8)
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, mkernel)
    filtered = cv2.dilate(filtered, mkernel)

    fk = max(5, int(min(h, w) * 0.018))
    fk += fk % 2 == 0
    occ = cv2.GaussianBlur(filtered.astype(np.float32), (fk, fk), fk * 0.35)
    occ = np.clip(occ, 0.0, 1.0)
    occ = np.minimum(occ, np.clip(face_mask * 1.15, 0.0, 1.0))
    occ *= strength
    return occ.astype(np.float32)


# ── Main swapper class ────────────────────────────────────────────────────────
class AdaptiveFaceSwapper:
    """
    Adaptive face swapper — works on any face regardless of size, angle, or lighting.
    Wraps InsightFace inswapper with:
      - Auto ONNX provider selection
      - Multi-detection ctx sizes (320 / 640) for small & large faces
      - Soft feathered paste-back
      - Occlusion preservation
      - Gender / similarity filtering
    """

    def __init__(self, models_path: str):
        self.models_path = models_path
        self.insightface_models_path = os.path.join(models_path, "insightface", "models")
        os.makedirs(self.insightface_models_path, exist_ok=True)

        self._is_linux = platform.system().lower() == "linux"
        self._providers = _pick_ort_providers(self._is_linux)
        self._lock = threading.RLock()

        self._analyser: Optional["FaceAnalysis"] = None
        self._swapper = None
        self._swapper_path: Optional[str] = None

        self.occlusion_enabled: bool = True
        self.occlusion_strength: float = 1.0
        self.occlusion_sensitivity: float = 0.55

        print(f"{TAG} Providers: {self._providers}")
        print(f"{TAG} InsightFace models path: {self.insightface_models_path}")

    # ── Initialization ────────────────────────────────────────────────────────
    def _ensure_analyser(self):
        if self._analyser is not None:
            return
        if not INSIGHTFACE_OK:
            raise RuntimeError("InsightFace not installed")
        print(f"{TAG} Loading face analyser...")

        # InsightFace 0.7.3 (Forge build) only accepts (name, root).
        # Provider selection is done via ctx_id in prepare():
        #   ctx_id = 0  → CUDA
        #   ctx_id = -1 → CPU
        use_cpu = "CUDAExecutionProvider" not in self._providers
        ctx_id = -1 if use_cpu else 0

        try:
            analyser = FaceAnalysis(
                name="buffalo_l",
                root=self.insightface_models_path,
            )
        except AssertionError as e:
            # Some InsightFace builds fail to classify buffalo_l detection model
            print(f"{TAG} buffalo_l unavailable ({e}); trying antelopev2")
            analyser = FaceAnalysis(
                name="antelopev2",
                root=self.insightface_models_path,
            )

        try:
            analyser.prepare(ctx_id=ctx_id, det_size=(640, 640))
        except Exception as prep_err:
            if ctx_id == 0:
                print(f"{TAG} CUDA prepare failed ({prep_err}); retrying on CPU")
                analyser = FaceAnalysis(
                    name="buffalo_l",
                    root=self.insightface_models_path,
                )
                analyser.prepare(ctx_id=-1, det_size=(640, 640))
            else:
                raise

        self._analyser = analyser
        print(f"{TAG} Face analyser ready (ctx_id={ctx_id})")

    def _ensure_swapper(self, swapper_name: str):
        # Search multiple locations (matching V3 behaviour)
        insightface_root = os.path.dirname(self.insightface_models_path)
        search_paths = [
            os.path.join(self.insightface_models_path, swapper_name),
            os.path.join(insightface_root, swapper_name),
            os.path.join(self.models_path, swapper_name),
            os.path.join(self.models_path, "reactor", swapper_name),
        ]

        swapper_path = None
        for p in search_paths:
            if os.path.exists(p):
                swapper_path = p
                break

        if self._swapper is not None and self._swapper_path == swapper_path:
            return  # Already loaded

        if swapper_path is None:
            raise FileNotFoundError(
                f"{TAG} Swapper model not found: {swapper_name}\n"
                f"  Searched: {search_paths}"
            )

        print(f"{TAG} Loading swapper: {swapper_name}")

        # InsightFace 0.7.3: model_zoo.get_model may or may not accept 'providers'.
        # V3 pattern: try with providers, catch TypeError, retry without.
        try:
            self._swapper = mz.get_model(swapper_path, providers=self._providers)
        except TypeError as e:
            if "providers" not in str(e):
                raise
            print(f"{TAG} model_zoo.get_model does not accept providers; using default")
            self._swapper = mz.get_model(swapper_path)

        self._swapper_path = swapper_path
        print(f"{TAG} Swapper ready: {swapper_path}")

    # ── Face detection (adaptive) ─────────────────────────────────────────────
    def _detect_faces_adaptive(self, img: np.ndarray) -> list:
        """
        Detect faces using two pass sizes.
        Pass 1: 640x640 (normal + large faces)
        Pass 2: 320x320 if pass 1 found nothing (small faces in high-res image)
        """
        self._ensure_analyser()
        faces = self._analyser.get(img)
        if not faces:
            # Retry with smaller det_size for small faces
            self._analyser.prepare(ctx_id=0, det_size=(320, 320))
            faces = self._analyser.get(img)
            self._analyser.prepare(ctx_id=0, det_size=(640, 640))
        return sorted(faces, key=lambda f: float(f.bbox[0]))  # left-to-right order

    # ── Gender helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _gender(face) -> str:
        try:
            g = int(face.gender) if hasattr(face, "gender") and face.gender is not None else -1
            return "M" if g == 1 else ("F" if g == 0 else "U")
        except Exception:
            return "U"

    def _filter_by_gender(self, faces: list, gender: str) -> list:
        if gender == "A":
            return faces
        return [f for f in faces if self._gender(f) == gender]

    # ── Core swap ────────────────────────────────────────────────────────────
    def swap_faces(
        self,
        source_img: np.ndarray,
        target_img: np.ndarray,
        source_face_idx: int = 0,
        target_face_idx: int = 0,
        swapper_model: str = "inswapper_128.onnx",
        gender_match: str = "S",
        swap_strength: float = 1.0,
    ) -> Tuple[np.ndarray, str]:
        """
        Swap one face from source onto target.
        Returns (result_bgr, status_message).

        Args:
            gender_match: 'S'=smart (same gender), 'A'=all, 'M'=male only, 'F'=female only
            swap_strength: 0.0=no swap, 1.0=full swap (alpha blend)
        """
        t0 = time.time()
        with self._lock:
            try:
                self._ensure_analyser()
                self._ensure_swapper(swapper_model)

                # ── Interrupt check ───────────────────────────────────
                if _is_interrupted():
                    print(f"{TAG} ✋ Interrupted before face detection")
                    return target_img.copy(), "Interrupted"

                # Critical diagnostic: are source and target actually different images?
                src_h, src_w = source_img.shape[:2]
                tgt_h, tgt_w = target_img.shape[:2]
                src_hash = hash(source_img.tobytes()[:4096])
                tgt_hash = hash(target_img.tobytes()[:4096])
                same_data = np.array_equal(source_img, target_img)
                print(f"{TAG} ▸ Source img: {src_w}×{src_h}, hash={src_hash}")
                print(f"{TAG} ▸ Target img: {tgt_w}×{tgt_h}, hash={tgt_hash}")
                print(f"{TAG} ▸ Same image? {same_data}")
                if same_data:
                    print(f"{TAG} ⚠ CRITICAL: source_img and target_img are IDENTICAL — swap will be a no-op!")

                # Detect faces
                src_faces = self._detect_faces_adaptive(source_img)
                if _is_interrupted():
                    print(f"{TAG} ✋ Interrupted after source detection")
                    return target_img.copy(), "Interrupted"
                tgt_faces = self._detect_faces_adaptive(target_img)

                if not src_faces:
                    return target_img.copy(), "Error: No face detected in source image"
                if not tgt_faces:
                    return target_img.copy(), "Error: No face detected in target image"

                # Pick source face
                src_face = src_faces[min(source_face_idx, len(src_faces) - 1)]
                src_gender = self._gender(src_face)

                # Filter target faces by gender
                if gender_match == "S":
                    filtered_tgt = self._filter_by_gender(tgt_faces, src_gender)
                    if not filtered_tgt:
                        print(f"{TAG} Smart gender match: no {src_gender} face in target, using all")
                        filtered_tgt = tgt_faces
                elif gender_match in ("M", "F"):
                    filtered_tgt = self._filter_by_gender(tgt_faces, gender_match)
                    if not filtered_tgt:
                        print(f"{TAG} Gender filter '{gender_match}': none found, using all")
                        filtered_tgt = tgt_faces
                else:
                    filtered_tgt = tgt_faces

                tgt_face = filtered_tgt[min(target_face_idx, len(filtered_tgt) - 1)]
                tgt_gender = self._gender(tgt_face)

                # Log face locations and embedding similarity
                src_bbox = [int(v) for v in src_face.bbox]
                tgt_bbox = [int(v) for v in tgt_face.bbox]
                print(f"{TAG} Source face bbox: {src_bbox}")
                print(f"{TAG} Target face bbox: {tgt_bbox}")

                # Embedding cosine similarity — if close to 1.0, faces look the same to inswapper
                src_emb = getattr(src_face, "normed_embedding", None)
                tgt_emb = getattr(tgt_face, "normed_embedding", None)
                if src_emb is not None and tgt_emb is not None:
                    cos_sim = float(np.dot(src_emb, tgt_emb))
                    print(f"{TAG} ▸ Embedding cosine similarity: {cos_sim:.4f}")
                    if cos_sim > 0.85:
                        print(f"{TAG} ⚠ Very high similarity ({cos_sim:.4f}) — faces look nearly identical to inswapper")
                else:
                    print(f"{TAG} ▸ Embeddings: src={'OK' if src_emb is not None else 'NONE'}, tgt={'OK' if tgt_emb is not None else 'NONE'}")

                print(f"{TAG} Swap: source({src_gender}) → target({tgt_gender})")

                # Save debug images so user can verify
                debug_dir = os.path.join(self.models_path, "..", "extensions", "Reactorv4", "_debug")
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, "source_received.png"), source_img)
                cv2.imwrite(os.path.join(debug_dir, "target_received.png"), target_img)
                print(f"{TAG} ▸ Debug images saved to {debug_dir}")

                # Perform swap
                result = target_img.copy()
                result = self._swapper.get(result, tgt_face, src_face, paste_back=True)

                # Diagnostic: verify swap actually changed pixels
                diff = cv2.absdiff(target_img, result)
                mean_diff = float(diff.mean())
                max_diff = float(diff.max())
                changed_pixels = int(np.count_nonzero(diff.sum(axis=2) > 5))
                total_pixels = target_img.shape[0] * target_img.shape[1]
                print(f"{TAG} ▸ Swap pixel diff: mean={mean_diff:.2f}, max={max_diff:.0f}, changed={changed_pixels}/{total_pixels} ({100*changed_pixels/total_pixels:.1f}%)")
                if mean_diff < 0.5:
                    print(f"{TAG} ⚠ WARNING: Swap produced near-identical image — inswapper may have failed silently")
                    cv2.imwrite(os.path.join(debug_dir, "swap_result.png"), result)
                    print(f"{TAG} ▸ Swap result saved to {debug_dir}/swap_result.png")

                # Strength blend: lerp between original target and swapped
                if swap_strength < 1.0:
                    result = cv2.addWeighted(
                        target_img, 1.0 - swap_strength,
                        result, swap_strength,
                        0.0,
                    )

                # Occlusion preservation (hair, glasses, hands)
                if self.occlusion_enabled:
                    face_mask, _ = _build_feathered_face_mask(tgt_face, target_img.shape)
                    occ = _detect_occlusions(
                        target_img, result, face_mask,
                        self.occlusion_sensitivity, self.occlusion_strength
                    )
                    if float(np.max(occ)) > 0.01:
                        occ_3ch = occ[:, :, None]
                        result = (
                            result.astype(np.float32) * (1.0 - occ_3ch) +
                            target_img.astype(np.float32) * occ_3ch
                        ).astype(np.uint8)

                elapsed = time.time() - t0
                msg = f"Swapped in {elapsed:.2f}s"
                print(f"{TAG} {msg}")
                return result, msg

            except Exception as exc:
                import traceback
                traceback.print_exc()
                return target_img.copy(), f"Error: {exc}"

    def swap_all_faces(
        self,
        source_imgs: List[np.ndarray],
        target_img: np.ndarray,
        swapper_model: str = "inswapper_128.onnx",
        gender_match: str = "S",
        swap_strength: float = 1.0,
        similarity_threshold: float = 0.30,
    ) -> Tuple[np.ndarray, str]:
        """
        Swap ALL detected faces in target using best-matching source face.
        Each target face is matched to the most similar source face by embedding cosine similarity.
        Falls back to index-0 source if no match found above threshold.
        """
        t0 = time.time()
        with self._lock:
            try:
                self._ensure_analyser()
                self._ensure_swapper(swapper_model)

                if _is_interrupted():
                    return target_img.copy(), "Interrupted"

                src_all = self._detect_faces_adaptive(source_imgs[0]) if len(source_imgs) == 1 else []
                for si in source_imgs:
                    detected = self._detect_faces_adaptive(si)
                    for f in detected:
                        if f not in src_all:
                            src_all.append(f)

                tgt_faces = self._detect_faces_adaptive(target_img)

                if not src_all:
                    return target_img.copy(), "Error: No faces in source"
                if not tgt_faces:
                    return target_img.copy(), "Error: No faces in target"

                result = target_img.copy()
                swapped_count = 0

                for tgt_face in tgt_faces:
                    if _is_interrupted():
                        print(f"{TAG} ✋ Interrupted during multi-face swap")
                        break
                    tgt_gender = self._gender(tgt_face)

                    # Find best matching source face
                    best_src = None
                    best_sim = -1.0
                    tgt_emb = getattr(tgt_face, "normed_embedding", None)

                    for sf in src_all:
                        sg = self._gender(sf)
                        if gender_match == "S" and sg != tgt_gender and sg != "U":
                            continue
                        if gender_match in ("M", "F") and sg != gender_match:
                            continue
                        src_emb = getattr(sf, "normed_embedding", None)
                        if tgt_emb is not None and src_emb is not None:
                            sim = float(np.dot(tgt_emb, src_emb))
                        else:
                            sim = similarity_threshold  # assume match
                        if sim > best_sim:
                            best_sim = sim
                            best_src = sf

                    if best_src is None or best_sim < similarity_threshold:
                        best_src = src_all[0]

                    before = result.copy()
                    result = self._swapper.get(result, tgt_face, best_src, paste_back=True)

                    if swap_strength < 1.0:
                        result = cv2.addWeighted(before, 1.0 - swap_strength, result, swap_strength, 0.0)

                    if self.occlusion_enabled:
                        face_mask, _ = _build_feathered_face_mask(tgt_face, target_img.shape)
                        occ = _detect_occlusions(
                            target_img, result, face_mask,
                            self.occlusion_sensitivity, self.occlusion_strength
                        )
                        if float(np.max(occ)) > 0.01:
                            occ_3ch = occ[:, :, None]
                            result = (
                                result.astype(np.float32) * (1.0 - occ_3ch) +
                                target_img.astype(np.float32) * occ_3ch
                            ).astype(np.uint8)

                    swapped_count += 1

                elapsed = time.time() - t0
                msg = f"Auto-swapped {swapped_count} face(s) in {elapsed:.2f}s"
                print(f"{TAG} {msg}")
                return result, msg

            except Exception as exc:
                import traceback
                traceback.print_exc()
                return target_img.copy(), f"Error: {exc}"

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def release(self):
        with self._lock:
            if self._swapper is not None:
                try:
                    if hasattr(self._swapper, "session"):
                        del self._swapper.session
                except Exception:
                    pass
                del self._swapper
                self._swapper = None
            if self._analyser is not None:
                try:
                    if hasattr(self._analyser, "models"):
                        for m in self._analyser.models.values():
                            if hasattr(m, "session"):
                                del m.session
                        self._analyser.models.clear()
                except Exception:
                    pass
                del self._analyser
                self._analyser = None
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        print(f"{TAG} Resources released")


# ── Singleton ─────────────────────────────────────────────────────────────────
_swapper_instance: Optional[AdaptiveFaceSwapper] = None
_swapper_lock = threading.Lock()


def get_swapper(models_path: str) -> AdaptiveFaceSwapper:
    global _swapper_instance
    with _swapper_lock:
        if _swapper_instance is None:
            _swapper_instance = AdaptiveFaceSwapper(models_path)
        return _swapper_instance
