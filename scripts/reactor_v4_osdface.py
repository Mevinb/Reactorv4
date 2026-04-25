"""
ReactorV4 — Multi-Model Safe Face Restorer

All models are ONNX from FaceFusion's official HuggingFace repo.
Source: https://huggingface.co/facefusion/models-3.0.0
Format: ONNX — no pickle, no arbitrary code execution, fully safe.

Available models (all auto-downloadable):
  restoreformer_plus_plus  — 🥇 Most natural skin, transformer-based, tight identity
  codeformer               — 🥈 Best overall fidelity, adjustable strength
  gfpgan_1.4               — Reliable GAN, good speed/quality balance
  gpen_bfr_512             — Sharp, already in V3 models dir
  gpen_bfr_1024            — Higher res GPEN
  gpen_bfr_2048            — Maximum sharpness (can over-process)

All models normalise: BGR → RGB, [-1, 1] input, [-1, 1] output  
(standard for all GAN-based face restorers)
"""

from __future__ import annotations

import gc
import hashlib
import os
import sys
import threading
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

TAG = "[ReactorV4 Restorer]"


def _is_interrupted() -> bool:
    try:
        from reactor_v4_pipeline import is_interrupted
        return is_interrupted()
    except ImportError:
        return False

HF_BASE = "https://huggingface.co/facefusion/models-3.0.0/resolve/main"

# ── Model registry ────────────────────────────────────────────────────────────
# name → (filename, size_px, size_mb, description)
RESTORER_REGISTRY: Dict[str, Tuple[str, int, str, str]] = {
    "restoreformer_plus_plus": (
        "restoreformer_plus_plus.onnx", 512, "~306 MB",
        "🥇 Most natural — Transformer, real skin texture, identity-safe"
    ),
    "codeformer": (
        "codeformer.onnx", 512, "~377 MB",
        "🥈 Best overall — Adjustable fidelity, widely tested"
    ),
    "gfpgan_1.4": (
        "gfpgan_1.4.onnx", 512, "~340 MB",
        "Reliable GAN — Good speed/quality balance"
    ),
    "gpen_bfr_512": (
        "gpen_bfr_512.onnx", 512, "~75 MB",
        "GPEN 512 — Sharp, already in V3 models dir"
    ),
    "gpen_bfr_1024": (
        "gpen_bfr_1024.onnx", 1024, "~285 MB",
        "GPEN 1024 — Higher resolution output"
    ),
    "gpen_bfr_2048": (
        "gpen_bfr_2048.onnx", 2048, "~286 MB",
        "GPEN 2048 — Maximum sharpness (can over-process)"
    ),
    "none": (
        "", 0, "0 MB",
        "No restoration — skip this step"
    ),
}

DEFAULT_MODEL = "restoreformer_plus_plus"


# ── ORT provider selection ────────────────────────────────────────────────────
try:
    import onnxruntime as ort
    ORT_OK = True
except ImportError:
    ORT_OK = False
    print(f"{TAG} WARNING: onnxruntime not installed")


def _ort_providers() -> List[str]:
    import platform
    is_linux = platform.system().lower() == "linux"
    force_cuda = os.environ.get("REACTOR_V4_ENABLE_CUDA", "0").strip().lower() in {"1", "true"}
    if is_linux and not force_cuda:
        return ["CPUExecutionProvider"]
    if torch.cuda.is_available():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


# ── Model path resolution ─────────────────────────────────────────────────────
def _model_dir(models_root: str) -> str:
    """ReactorV4's dedicated model directory inside the WebUI models folder."""
    d = os.path.join(models_root, "reactor_v4", "face_restorers")
    os.makedirs(d, exist_ok=True)
    return d


def _find_model(models_root: str, filename: str) -> Optional[str]:
    """Search several locations for an ONNX model file."""
    candidates = [
        os.path.join(_model_dir(models_root), filename),
        os.path.join(models_root, "facerestore_models", filename),
        os.path.join(models_root, filename),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


# ── Auto-downloader ───────────────────────────────────────────────────────────
def _download_model(models_root: str, filename: str) -> Optional[str]:
    """Download a model from FaceFusion HuggingFace to the V4 model dir."""
    dest = os.path.join(_model_dir(models_root), filename)
    if os.path.isfile(dest):
        return dest
    url = f"{HF_BASE}/{filename}"
    print(f"{TAG} Downloading {filename} ...")
    print(f"{TAG} Source: {url}")
    print(f"{TAG} Destination: {dest}")
    try:
        def _reporthook(count, block_size, total_size):
            if total_size > 0 and count % 200 == 0:
                pct = min(100, count * block_size * 100 // total_size)
                print(f"{TAG}   ... {pct}%", end="\r")
        urllib.request.urlretrieve(url, dest, _reporthook)
        print(f"\n{TAG} Download complete: {filename}")
        return dest
    except Exception as exc:
        print(f"\n{TAG} Download failed: {exc}")
        if os.path.isfile(dest):
            os.remove(dest)
        return None


# ── Fallback enhancer (no model) ──────────────────────────────────────────────
def _fallback_enhance(face_bgr: np.ndarray) -> np.ndarray:
    """CLAHE + unsharp mask when no ONNX model is available."""
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.8)
    return np.clip(cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0), 0, 255).astype(np.uint8)


# ── Core ONNX inference ───────────────────────────────────────────────────────
def _run_onnx_session(
    session: "ort.InferenceSession",
    input_name: str,
    output_name: str,
    face_bgr: np.ndarray,
    model_size: int,
    model_key: str,
) -> np.ndarray:
    """
    Unified ONNX inference for all face restoration models.
    All models use the same normalisation convention:
      Input:  RGB float32 [1,3,H,W] in [-1, 1]
      Output: RGB float32 [1,3,H,W] in [-1, 1]
    CodeFormer has a fidelity input — we pass 0.5 as a second input if needed.
    """
    oh, ow = face_bgr.shape[:2]
    if oh != model_size or ow != model_size:
        resized = cv2.resize(face_bgr, (model_size, model_size), interpolation=cv2.INTER_LANCZOS4)
    else:
        resized = face_bgr.copy()

    # BGR → RGB, normalise [-1, 1]
    rgb = resized[:, :, ::-1].astype(np.float32) / 127.5 - 1.0
    inp = rgb.transpose(2, 0, 1)[None].astype(np.float32)  # [1,3,H,W]

    # Build feed dict
    feeds = {input_name: inp}
    # CodeFormer has a second input: fidelity weight
    if model_key == "codeformer":
        try:
            inputs = session.get_inputs()
            if len(inputs) > 1:
                fidelity_name = inputs[1].name
                feeds[fidelity_name] = np.array([0.5], dtype=np.float64)
        except Exception:
            pass

    out = session.run([output_name], feeds)[0]  # [1,3,H,W]
    out = np.squeeze(out, 0).transpose(1, 2, 0)  # [H,W,3]
    out_bgr = np.clip((out + 1.0) * 127.5, 0, 255).astype(np.uint8)[:, :, ::-1]

    if oh != model_size or ow != model_size:
        out_bgr = cv2.resize(out_bgr, (ow, oh), interpolation=cv2.INTER_AREA)
    return out_bgr


# ── Main restorer class ───────────────────────────────────────────────────────
class SafeFaceRestorer:
    """
    Multi-model safe face restorer using ONNX Runtime only.

    Supports RestoreFormer++, CodeFormer, GFPGANv1.4, GPEN family.
    All models auto-downloaded from FaceFusion's trusted HuggingFace repo.
    No .pth / pickle / arbitrary code execution ever.
    """

    def __init__(self, models_root: str, model_key: str = DEFAULT_MODEL):
        self.models_root  = models_root
        self.model_key    = model_key
        self._lock        = threading.Lock()
        self._session: Optional["ort.InferenceSession"] = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self._loaded_key: Optional[str] = None
        self._model_size: int = 512
        self._face_helper = None
        self._face_helper_size: int = 0
        self._tried: bool = False

        print(f"{TAG} Initialised — model: {model_key}")

    # ── Lazy loading ──────────────────────────────────────────────────────────
    def _load_model(self, key: str) -> bool:
        """Load (or re-load) the ONNX session for the given model key."""
        if key == "none":
            self._session = None
            self._loaded_key = "none"
            return True

        if key not in RESTORER_REGISTRY:
            print(f"{TAG} Unknown model key: {key}")
            return False

        filename, size_px, size_str, desc = RESTORER_REGISTRY[key]
        print(f"{TAG} Loading: {key} ({desc}, {size_str})")

        path = _find_model(self.models_root, filename)
        if path is None:
            print(f"{TAG} Not found locally — downloading from FaceFusion HuggingFace...")
            path = _download_model(self.models_root, filename)

        if path is None:
            print(f"{TAG} ✗ Could not obtain {filename} — using fallback")
            return False

        if not ORT_OK:
            print(f"{TAG} onnxruntime not available — using fallback")
            return False

        try:
            providers = _ort_providers()
            sess = ort.InferenceSession(path, providers=providers)
            self._session    = sess
            self._input_name = sess.get_inputs()[0].name
            self._output_name = sess.get_outputs()[0].name
            self._model_size = size_px
            self._loaded_key = key
            print(f"{TAG} ✓ {key} ready ({size_px}px, providers={providers})")
            return True
        except Exception as exc:
            print(f"{TAG} Session error: {exc}")
            import traceback; traceback.print_exc()
            return False

    def _ensure_loaded(self):
        if self._loaded_key != self.model_key:
            if self._session is not None:
                del self._session
                self._session = None
            self._load_model(self.model_key)
            # Rebuild face helper if size changed
            if self._model_size != self._face_helper_size:
                self._face_helper = None

    # ── FaceRestoreHelper ─────────────────────────────────────────────────────
    def _ensure_face_helper(self):
        if self._face_helper is not None:
            return
        size = self._model_size if self._model_size > 0 else 512
        try:
            webui_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            if webui_root not in sys.path:
                sys.path.insert(0, webui_root)
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
            from facexlib.detection import retinaface
            from modules import devices
            dev = devices.device_codeformer if torch.cuda.is_available() else devices.cpu
            if hasattr(retinaface, "device"):
                retinaface.device = dev
            self._face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=size,
                crop_ratio=(1, 1),
                det_model="retinaface_resnet50",
                save_ext="png",
                use_parse=True,
                device=dev,
            )
            self._face_helper_size = size
            print(f"{TAG} FaceRestoreHelper ready (face_size={size})")
        except Exception as exc:
            print(f"{TAG} FaceRestoreHelper unavailable: {exc}")
            self._face_helper = "unavailable"

    # ── Public API ────────────────────────────────────────────────────────────
    def set_model(self, key: str):
        """Switch restorer model (call from UI dropdown)."""
        with self._lock:
            if key != self.model_key:
                self.model_key = key
                self._tried = False
                print(f"{TAG} Model switched → {key}")

    def restore(self, img_bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Restore faces in a full image.
        strength: 1.0 = full restoration, 0.5 = blend 50/50 with original.
        """
        t0 = time.time()
        with self._lock:
            self._ensure_loaded()
            self._ensure_face_helper()

            if self.model_key == "none":
                return img_bgr

            h, w = img_bgr.shape[:2]
            print(f"{TAG} Restoring {w}×{h} with [{self.model_key}]...")

            helper = self._face_helper
            if helper == "unavailable":
                restored = self._run_on_face(img_bgr)
                if strength < 1.0:
                    restored = cv2.addWeighted(img_bgr, 1.0 - strength, restored, strength, 0)
                return restored

            helper.clean_all()
            helper.read_image(img_bgr)
            helper.get_face_landmarks_5()
            helper.align_warp_face()

            if not helper.cropped_faces:
                print(f"{TAG} No faces detected — returning original")
                helper.clean_all()
                return img_bgr

            n = len(helper.cropped_faces)
            print(f"{TAG} {n} face(s) found")
            helper.restored_faces = []

            for i, crop in enumerate(helper.cropped_faces):
                if _is_interrupted():
                    print(f"{TAG} ✋ Interrupted during face restoration")
                    helper.clean_all()
                    return img_bgr
                t_f = time.time()
                try:
                    restored_face = self._run_on_face(crop)
                    if strength < 1.0:
                        restored_face = cv2.addWeighted(
                            crop, 1.0 - strength, restored_face, strength, 0
                        )
                except Exception as exc:
                    print(f"{TAG} Face {i+1} error: {exc}")
                    restored_face = _fallback_enhance(crop)
                print(f"{TAG} Face {i+1}/{n} done in {time.time()-t_f:.3f}s")
                helper.restored_faces.append(restored_face)

            helper.get_inverse_affine(None)
            result = helper.paste_faces_to_input_image()
            helper.clean_all()

            print(f"{TAG} Done in {time.time()-t0:.3f}s")
            gc.collect()
            return result

    def _run_on_face(self, face_bgr: np.ndarray) -> np.ndarray:
        if self._session is None:
            return _fallback_enhance(face_bgr)
        return _run_onnx_session(
            self._session,
            self._input_name,
            self._output_name,
            face_bgr,
            self._model_size,
            self.model_key,
        )

    def get_status(self) -> str:
        """Human-readable status for the UI."""
        if self.model_key == "none":
            return "⏭ Restoration disabled"
        _, _, size_str, desc = RESTORER_REGISTRY.get(self.model_key, ("", 0, "?", "Unknown"))
        path = None
        if self.model_key in RESTORER_REGISTRY:
            fname = RESTORER_REGISTRY[self.model_key][0]
            path = _find_model(self.models_root, fname)
        if self._session is not None and self._loaded_key == self.model_key:
            return f"✅ {self.model_key} loaded ({size_str}) — {desc}"
        elif path:
            return f"⚙️ {self.model_key} found locally (not loaded yet) — {desc}"
        else:
            return f"📥 {self.model_key} will auto-download on first use ({size_str}) — {desc}"

    def release(self):
        with self._lock:
            if self._session is not None:
                del self._session
                self._session = None
            if self._face_helper not in (None, "unavailable"):
                try: self._face_helper.clean_all()
                except Exception: pass
                self._face_helper = None
            self._loaded_key = None
        gc.collect()
        if torch.cuda.is_available():
            try: torch.cuda.empty_cache()
            except Exception: pass
        print(f"{TAG} Released")


# ── Singleton ─────────────────────────────────────────────────────────────────
_instance: Optional[SafeFaceRestorer] = None
_lock = threading.Lock()


def get_safe_restorer(models_root: str, model_key: str = DEFAULT_MODEL) -> SafeFaceRestorer:
    global _instance
    with _lock:
        if _instance is None:
            _instance = SafeFaceRestorer(models_root, model_key)
        else:
            _instance.set_model(model_key)
        return _instance


def clear_restorer_cache():
    global _instance
    with _lock:
        if _instance is not None:
            _instance.release()
            _instance = None


def list_restorer_choices() -> List[Tuple[str, str]]:
    """Return (label, key) pairs for the Gradio dropdown."""
    choices = []
    for key, (_, px, size_str, desc) in RESTORER_REGISTRY.items():
        if key == "none":
            choices.append(("⏭ None (skip restoration)", "none"))
        else:
            label = f"{key.replace('_', ' ').title()} — {desc.split('—')[0].strip()} ({size_str})"
            choices.append((label, key))
    return choices
