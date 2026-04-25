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

from reactor_v4_swapper       import AdaptiveFaceSwapper, get_swapper
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

                # Step 1 — Swap
                print(f"\n{TAG} Step 1/3 — Adaptive Face Swap")
                swapper = self._get_swapper()
                result, swap_msg = swapper.swap_faces(
                    source_img, target_img,
                    source_face_idx, target_face_idx,
                    swapper_model, self.gender_match, self.swap_strength,
                )
                if "Error" in swap_msg:
                    return target_img.copy(), swap_msg

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

                # Step 3 — Texture Transfer
                if self.enable_texture and self.texture_strength > 0.0:
                    print(f"\n{TAG} Step 3/3 — E4S Texture Transfer")
                    texture = self._get_texture()
                    bbox = self._face_bbox(swapper, result, target_face_idx)
                    result = texture.transfer(
                        result, source_img,
                        face_bbox=bbox,
                        strength=self.texture_strength,
                        hf_strength=self.hf_strength,
                        protect_non_skin=True,
                    )
                else:
                    print(f"{TAG} Step 3/3 — Texture Transfer SKIPPED")

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
