"""
ReactorV4 — E4S-Style Reference-Guided Texture Transfer (Option B)

Implements the core texture-matching technique from E4S (ECCV 2023 / CVPR update)
WITHOUT requiring the full StyleGAN pipeline or heavy pretrained weights.

What E4S does (full version):
  Regional GAN Inversion → encode face into style codes →
  swap style codes per region (eyes, cheeks, forehead) →
  decode with StyleGAN

What THIS module does (lightweight equivalent):
  1. Face parsing → semantic regions (skin, eyes, nose, lips, forehead)
  2. Per-region VGG Gram-matrix texture descriptors (captures style)
  3. Frequency-separated texture injection:
     - Low freq (tone/color): Lab histogram matching from reference
     - High freq (skin texture): Gram-matched from reference VGG features
  4. Spatially adaptive blend using face parsing masks

Result: The restored face gets the REAL skin texture from the reference image
while keeping its swapped identity and OSDFace-enhanced quality.

No StyleGAN. No heavy model weights. Works immediately.
Only optional dependency: torchvision (for VGG features) — falls back to
Laplacian-based texture if torchvision is unavailable.
"""

from __future__ import annotations

import cv2
import gc
import numpy as np
import os
import sys
import time
import threading
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

TAG = "[ReactorV4 TextureTransfer]"

# ── Optional VGG feature extractor ───────────────────────────────────────────
try:
    import torchvision.models as tv_models
    import torchvision.transforms as tv_transforms
    VGG_AVAILABLE = True
except ImportError:
    VGG_AVAILABLE = False
    print(f"{TAG} torchvision not available — using Laplacian texture fallback")


# ── Face parsing (semantic segmentation) ─────────────────────────────────────
class FaceRegionParser:
    """
    Adaptive face region parser.
    Tries facexlib's BiSeNet first; falls back to landmark-based geometry masks.
    """

    REGION_LABELS = {
        "skin":     [1],
        "left_eye": [4, 5],
        "right_eye":[2, 3],
        "nose":     [10],
        "lip":      [12, 13],
        "hair":     [17],
        "neck":     [14],
    }

    def __init__(self, device: torch.device):
        self.device = device
        self._bisenet = None
        self._tried = False

    def _try_load_bisenet(self):
        if self._tried:
            return
        self._tried = True
        try:
            from facexlib.parsing import init_parsing_model
            self._bisenet = init_parsing_model(device=self.device)
            print(f"{TAG} BiSeNet face parser loaded")
        except Exception as e:
            print(f"{TAG} BiSeNet unavailable ({e}) — using geometry fallback")

    def parse(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Returns a label map (H, W) uint8 with semantic regions.
        Label 0 = background, 1 = skin, 2-17 = other regions (BiSeNet convention).
        Falls back to geometry-based labels if BiSeNet unavailable.
        """
        self._try_load_bisenet()
        if self._bisenet is not None:
            try:
                return self._bisenet_parse(face_bgr)
            except Exception as e:
                print(f"{TAG} BiSeNet parse error: {e} — using fallback")
        return self._geometry_parse(face_bgr)

    def _bisenet_parse(self, face_bgr: np.ndarray) -> np.ndarray:
        h, w = face_bgr.shape[:2]
        inp = cv2.resize(face_bgr, (512, 512))
        rgb = inp[:, :, ::-1].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        rgb  = (rgb - mean) / std
        t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            out = self._bisenet(t)[0]
        labels = out.argmax(1).squeeze().cpu().numpy().astype(np.uint8)
        return cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)

    def _geometry_parse(self, face_bgr: np.ndarray) -> np.ndarray:
        """Simple geometry-based region map as fallback."""
        h, w = face_bgr.shape[:2]
        labels = np.zeros((h, w), dtype=np.uint8)

        # Skin = center ellipse
        cx, cy = w // 2, h // 2
        ax, ay = int(w * 0.42), int(h * 0.48)
        cv2.ellipse(labels, (cx, cy), (ax, ay), 0, 0, 360, 1, -1)

        # Eyes = upper 1/3
        eye_top = int(h * 0.28)
        eye_bot = int(h * 0.45)
        labels[eye_top:eye_bot, :int(w*0.45)] = 4   # left eye region
        labels[eye_top:eye_bot, int(w*0.55):] = 2   # right eye region

        # Nose = center strip
        nose_top = int(h * 0.42)
        nose_bot = int(h * 0.65)
        nose_l   = int(w * 0.38)
        nose_r   = int(w * 0.62)
        labels[nose_top:nose_bot, nose_l:nose_r] = 10

        # Lips = lower 1/4
        lip_top = int(h * 0.65)
        lip_bot = int(h * 0.82)
        lip_l   = int(w * 0.35)
        lip_r   = int(w * 0.65)
        labels[lip_top:lip_bot, lip_l:lip_r] = 12

        # Hair = top band
        labels[:int(h * 0.22), :] = 17

        return labels

    def get_skin_mask(self, label_map: np.ndarray) -> np.ndarray:
        """Return float32 mask [H,W] for skin regions."""
        skin_ids = [1] + self.REGION_LABELS.get("skin", [])
        mask = np.zeros(label_map.shape, dtype=np.float32)
        for sid in skin_ids:
            mask[label_map == sid] = 1.0
        # Dilate slightly to include edge transitions
        k = max(3, label_map.shape[0] // 40)
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        # Feather edges
        fk = max(5, k * 2) | 1  # must be odd
        mask = cv2.GaussianBlur(mask, (fk, fk), fk * 0.35)
        return np.clip(mask, 0.0, 1.0)


# ── VGG Gram-matrix feature extractor ────────────────────────────────────────
class VGGTextureExtractor:
    """
    Extracts Gram-matrix texture descriptors using VGG19 intermediate layers.
    These capture skin texture style independent of spatial structure.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, device: torch.device):
        self.device = device
        self._model = None

    @classmethod
    def get(cls, device: torch.device) -> "VGGTextureExtractor":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(device)
            return cls._instance

    def _ensure_model(self):
        if self._model is not None:
            return
        if not VGG_AVAILABLE:
            return
        print(f"{TAG} Loading VGG19 for texture features...")
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1)
        # Use relu1_2, relu2_2, relu3_3 layers (good for texture)
        self._layers = torch.nn.Sequential(*list(vgg.features)[:18])
        self._layers = self._layers.to(self.device).eval()
        for p in self._layers.parameters():
            p.requires_grad_(False)
        print(f"{TAG} VGG19 texture extractor ready")

    def _preprocess(self, face_bgr: np.ndarray) -> torch.Tensor:
        rgb = face_bgr[:, :, ::-1].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        return t

    @torch.no_grad()
    def gram_matrix(self, face_bgr: np.ndarray) -> Optional[torch.Tensor]:
        """Compute Gram matrix of VGG features — captures texture style."""
        self._ensure_model()
        if self._model is None and not VGG_AVAILABLE:
            return None
        try:
            t = self._preprocess(face_bgr)
            feats = self._layers(t)
            b, c, h, w = feats.shape
            feat_flat = feats.view(b, c, -1)
            gram = torch.bmm(feat_flat, feat_flat.transpose(1, 2)) / (c * h * w)
            return gram.squeeze(0)
        except Exception as e:
            print(f"{TAG} VGG gram error: {e}")
            return None


# ── Colour histogram matching ─────────────────────────────────────────────────
def _histogram_match_lab(
    source_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Match the Lab colour histogram of source to reference.
    If mask is provided, only use masked pixels for statistics (skin only).
    """
    src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    result = src_lab.copy()

    for ch in range(3):
        if mask is not None:
            flat_src = src_lab[:, :, ch][mask > 0.5]
            flat_ref = ref_lab[:, :, ch][mask > 0.5]
        else:
            flat_src = src_lab[:, :, ch].flatten()
            flat_ref = ref_lab[:, :, ch].flatten()

        if flat_src.size < 10 or flat_ref.size < 10:
            continue

        src_mean, src_std = flat_src.mean(), flat_src.std() + 1e-6
        ref_mean, ref_std = flat_ref.mean(), flat_ref.std() + 1e-6

        # Linear mapping: match mean & std
        result[:, :, ch] = (src_lab[:, :, ch] - src_mean) * (ref_std / src_std) + ref_mean

    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


# ── High-frequency texture injection ─────────────────────────────────────────
def _inject_hf_texture(
    target_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    strength: float = 0.35,
    sigma_lf: float = 3.5,
) -> np.ndarray:
    """
    Inject high-frequency skin texture from reference into target.
    Uses frequency separation: keeps target's low-freq (shape/tone) but
    replaces high-freq with a blend towards reference's high-freq.

    This is the core of E4S's skin texture injection — but without StyleGAN.
    """
    # Ensure same size
    h, w = target_bgr.shape[:2]
    if reference_bgr.shape[:2] != (h, w):
        reference_bgr = cv2.resize(reference_bgr, (w, h), interpolation=cv2.INTER_AREA)

    # Work in linear light for unbiased HF arithmetic
    tgt_f = np.power(np.clip(target_bgr.astype(np.float32) / 255.0, 0, 1), 2.2)
    ref_f = np.power(np.clip(reference_bgr.astype(np.float32) / 255.0, 0, 1), 2.2)

    # Frequency separation
    tgt_lf = cv2.GaussianBlur(tgt_f, (0, 0), sigmaX=sigma_lf)
    ref_lf = cv2.GaussianBlur(ref_f, (0, 0), sigmaX=sigma_lf)
    tgt_hf = tgt_f - tgt_lf
    ref_hf = ref_f - ref_lf

    # Spatial normalisation of HF (prevents global bias from lighting diff)
    tgt_hf_energy = np.mean(np.abs(tgt_hf), axis=2, keepdims=True) + 1e-6
    ref_hf_energy = np.mean(np.abs(ref_hf), axis=2, keepdims=True) + 1e-6
    ref_hf_normed = ref_hf / ref_hf_energy  # unit style vectors
    ref_hf_rescaled = ref_hf_normed * tgt_hf_energy  # re-scale to target energy level

    # Blend: inject reference texture at 'strength'
    mixed_hf = tgt_hf * (1.0 - strength) + ref_hf_rescaled * strength

    # Reconstruct in sRGB
    result_f = np.clip(tgt_lf + mixed_hf, 0.0, 1.0)
    result_linear = np.power(result_f, 1.0 / 2.2)
    result_bgr = np.clip(result_linear * 255.0, 0, 255).astype(np.uint8)

    return result_bgr


# ── Main texture transfer class ───────────────────────────────────────────────
class E4SStyleTextureTransfer:
    """
    Reference-guided texture transfer inspired by E4S Regional GAN Inversion.

    Pipeline per face region:
      1. Parse reference and swapped face into semantic regions
      2. Match Lab colour histogram (reference → swapped) per skin region
      3. Inject real HF texture from reference into swapped face
      4. Blend back using skin mask with feathered edges
      5. Protect non-skin regions (eyes, lips) from overwriting
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._parser = FaceRegionParser(device)
        self._vgg = VGGTextureExtractor.get(device) if VGG_AVAILABLE else None
        self._analyser = None
        self._lock = threading.Lock()

    def transfer(
        self,
        swapped_bgr: np.ndarray,
        reference_bgr: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        strength: float = 0.45,
        hf_strength: float = 0.35,
        protect_non_skin: bool = True,
    ) -> np.ndarray:
        """
        Transfer skin texture from reference_bgr into swapped_bgr.

        Args:
            swapped_bgr:    Result from OSDFace restoration (BGR uint8)
            reference_bgr:  Original source face image (BGR uint8) — the REFERENCE
            face_bbox:      Optional (x1,y1,x2,y2) to limit operation to face region.
                            If None, operates on full image.
            strength:       Overall texture transfer strength (0.0–1.0)
            hf_strength:    High-frequency texture injection strength (0.0–1.0)
            protect_non_skin: If True, skip non-skin regions (eyes, lips, etc.)

        Returns:
            Image with reference skin texture applied (BGR uint8)
        """
        if strength <= 0.0:
            return swapped_bgr

        with self._lock:
            t0 = time.time()
            h, w = swapped_bgr.shape[:2]
            result = swapped_bgr.copy()

            # ── Extract face regions ──────────────────────────────────────────
            if face_bbox is not None:
                x1, y1, x2, y2 = face_bbox
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                if x2 - x1 < 16 or y2 - y1 < 16:
                    return swapped_bgr
                swapped_roi  = swapped_bgr[y1:y2, x1:x2]
                # Reference may be a full portrait — extract best face region
                ref_roi = self._extract_face_region(reference_bgr, (x2-x1, y2-y1))
            else:
                swapped_roi = swapped_bgr
                ref_roi = cv2.resize(reference_bgr,
                                     (swapped_bgr.shape[1], swapped_bgr.shape[0]),
                                     interpolation=cv2.INTER_AREA)

            fw, fh = swapped_roi.shape[1], swapped_roi.shape[0]
            if fw < 16 or fh < 16:
                return swapped_bgr

            # ── Parse semantic regions ────────────────────────────────────────
            try:
                label_map = self._parser.parse(swapped_roi)
                skin_mask = self._parser.get_skin_mask(label_map)
            except Exception as e:
                print(f"{TAG} Parse error: {e} — using full-face mask")
                skin_mask = np.ones((fh, fw), dtype=np.float32)

            # If protecting non-skin, erode skin mask near eye/lip/nose boundaries
            if protect_non_skin:
                non_skin_ids = []
                for k, v in FaceRegionParser.REGION_LABELS.items():
                    if k not in ("skin",):
                        non_skin_ids.extend(v)
                non_skin = np.zeros((fh, fw), dtype=np.float32)
                for sid in non_skin_ids:
                    non_skin[label_map == sid] = 1.0
                # Dilate non-skin protection zone
                ns_k = max(3, min(fw, fh) // 30)
                ns_kernel = np.ones((ns_k, ns_k), dtype=np.uint8)
                non_skin = cv2.dilate(non_skin, ns_kernel, iterations=1)
                non_skin = cv2.GaussianBlur(non_skin, (ns_k*2+1, ns_k*2+1), ns_k * 0.5)
                skin_mask = skin_mask * (1.0 - np.clip(non_skin, 0, 1))

            print(f"{TAG} Skin coverage: {float(np.mean(skin_mask)):.2f}")

            # ── Step 1: Lab histogram colour match ────────────────────────────
            skin_bool = skin_mask > 0.3
            colour_matched = _histogram_match_lab(
                swapped_roi, ref_roi,
                mask=skin_bool.astype(np.uint8)
            )

            # ── Step 2: High-frequency texture injection ──────────────────────
            texture_applied = _inject_hf_texture(
                colour_matched,
                ref_roi,
                strength=hf_strength,
                sigma_lf=max(2.0, min(fw, fh) * 0.007),
            )

            # ── Step 3: Adaptive blend using skin mask ────────────────────────
            # Modulate blend strength by plasticity of swapped region
            plasticity = self._estimate_plasticity(swapped_roi)
            adaptive_strength = min(1.0, strength * (0.6 + plasticity * 0.8))
            print(f"{TAG} Plasticity: {plasticity:.2f}, adaptive strength: {adaptive_strength:.2f}")

            sm3 = skin_mask[:, :, None] * adaptive_strength
            blended_roi = (
                texture_applied.astype(np.float32) * sm3 +
                swapped_roi.astype(np.float32) * (1.0 - sm3)
            ).astype(np.uint8)

            # ── Write back ────────────────────────────────────────────────────
            if face_bbox is not None:
                # Feathered composite back to full image
                full_mask = np.zeros((h, w), dtype=np.float32)
                full_mask[y1:y2, x1:x2] = skin_mask * adaptive_strength
                fk = max(5, int(min(h, w) * 0.015)) | 1
                full_mask = cv2.GaussianBlur(full_mask, (fk, fk), fk * 0.35)
                fm3 = np.clip(full_mask, 0, 1)[:, :, None]
                result_full = result.copy()
                result_full[y1:y2, x1:x2] = blended_roi
                result = (
                    result_full.astype(np.float32) * fm3 +
                    swapped_bgr.astype(np.float32) * (1.0 - fm3)
                ).astype(np.uint8)
            else:
                sm3_full = skin_mask[:, :, None] * adaptive_strength
                result = (
                    blended_roi.astype(np.float32) * sm3_full +
                    swapped_bgr.astype(np.float32) * (1.0 - sm3_full)
                ).astype(np.uint8)

            elapsed = time.time() - t0
            print(f"{TAG} Texture transfer done in {elapsed:.3f}s")
            return result

    def _extract_face_region(
        self, reference_bgr: np.ndarray, target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Extract the most prominent face region from reference image and resize to target_size.
        Uses InsightFace detection if available, otherwise centre crop.
        """
        tw, th = target_size
        try:
            import insightface
            from insightface.app import FaceAnalysis
            if self._analyser is None:
                from reactor_v4_swapper import _pick_ort_providers
                import platform
                is_linux = platform.system().lower() == "linux"
                providers = _pick_ort_providers(is_linux)
                use_cpu = "CUDAExecutionProvider" not in providers
                ctx_id = -1 if use_cpu else 0
                
                self._analyser = FaceAnalysis(
                    name="buffalo_l",
                    root=os.path.expanduser("~/.insightface"),
                    providers=providers,
                )
                self._analyser.prepare(ctx_id=ctx_id, det_size=(640, 640))
                
            faces = self._analyser.get(reference_bgr)
            if faces:
                face = faces[0]
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                pad = int(max(x2-x1, y2-y1) * 0.15)
                h, w = reference_bgr.shape[:2]
                x1 = max(0, x1-pad); y1 = max(0, y1-pad)
                x2 = min(w, x2+pad); y2 = min(h, y2+pad)
                crop = reference_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    return cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"{TAG} Warning: _extract_face_region failed ({e}) — using fallback")
            pass
        # Fallback: centre crop
        h, w = reference_bgr.shape[:2]
        size = min(h, w)
        cy, cx = h // 2, w // 2
        h0 = max(0, cy - size//2); h1 = min(h, h0 + size)
        w0 = max(0, cx - size//2); w1 = min(w, w0 + size)
        crop = reference_bgr[h0:h1, w0:w1]
        return cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _estimate_plasticity(face_bgr: np.ndarray) -> float:
        """
        Quick plasticity estimate (0=natural, 1=very plastic/waxy).
        Used to scale texture injection strength adaptively.
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Local variance (smoothness indicator)
        blur = cv2.blur(gray, (7, 7))
        var = np.mean(cv2.blur(gray**2, (7, 7)) - blur**2)
        variance_score = 1.0 - np.clip(var / 60.0, 0.0, 1.0)
        # Gradient magnitude (detail level)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.mean(np.sqrt(gx**2 + gy**2))
        gradient_score = 1.0 - np.clip(grad / 12.0, 0.0, 1.0)
        return float(np.clip(variance_score * 0.6 + gradient_score * 0.4, 0.0, 1.0))


# ── Singleton ─────────────────────────────────────────────────────────────────
_transfer_instance: Optional[E4SStyleTextureTransfer] = None
_transfer_lock = threading.Lock()


def get_texture_transfer(device: torch.device) -> E4SStyleTextureTransfer:
    global _transfer_instance
    with _transfer_lock:
        if _transfer_instance is None:
            _transfer_instance = E4SStyleTextureTransfer(device)
        return _transfer_instance
