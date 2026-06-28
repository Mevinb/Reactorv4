"""
ReactorV4 — SD WebUI Forge UI

Pipeline: inswapper → RestoreFormer++ / CodeFormer / GFPGAN / GPEN (ONNX) → E4S Texture
"""

def create_script():
    try:
        import gradio as gr
        import modules.scripts as scripts
        from PIL import Image
        import numpy as np
        import cv2
        import os
        import sys

        _dir = os.path.dirname(os.path.abspath(__file__))
        if _dir not in sys.path:
            sys.path.insert(0, _dir)

        TAG = "[ReactorV4]"

        def _webui_root():
            return os.path.abspath(os.path.join(_dir, "..", "..", ".."))

        def _models_path():
            return os.path.join(_webui_root(), "models")

        def _swapper_choices():
            mdir = os.path.join(_models_path(), "insightface", "models")
            choices = []
            if os.path.isdir(mdir):
                for f in sorted(os.listdir(mdir)):
                    if f.endswith(".onnx"):
                        tag = ""
                        if "128" in f: tag = " (128px, Fast)"
                        elif "256" in f: tag = " (256px, HQ)"
                        choices.append((f + tag, f))
            if not choices:
                choices = [
                    ("inswapper_128.onnx (128px, Fast)", "inswapper_128.onnx"),
                    ("reswapper_256.onnx (256px, HQ)", "reswapper_256.onnx"),
                ]
            return choices

        def _restorer_choices():
            from reactor_v4_osdface import list_restorer_choices, RESTORER_REGISTRY, _find_model
            mp = _models_path()
            result = []
            for label, key in list_restorer_choices():
                if key != "none":
                    fname = RESTORER_REGISTRY[key][0]
                    found = _find_model(mp, fname)
                    indicator = " ✅" if found else " 📥"
                    result.append((label + indicator, key))
                else:
                    result.append((label, key))
            return result

        def _restorer_status(key: str) -> str:
            try:
                from reactor_v4_osdface import RESTORER_REGISTRY, _find_model
                mp = _models_path()
                if key == "none":
                    return "⏭ Restoration disabled — skipping this step"
                if key not in RESTORER_REGISTRY:
                    return f"⚠ Unknown model: {key}"
                fname, px, size_str, desc = RESTORER_REGISTRY[key]
                found = _find_model(mp, fname)
                dest = os.path.join(mp, "reactor_v4", "face_restorers", fname)
                if found:
                    return f"✅ **{key}** found at `{found}` — will load on first use"
                else:
                    return (
                        f"📥 **{key}** not yet downloaded ({size_str})\n"
                        f"Will auto-download from FaceFusion HuggingFace on first use.\n"
                        f"Destination: `{dest}`"
                    )
            except Exception as e:
                return f"⚠ Status check failed: {e}"

        # ── Script class ──────────────────────────────────────────────────────
        class ReactorV4Script(scripts.Script):

            def title(self):
                return "ReActor V4 — Multi-Model + E4S Texture"

            def show(self, is_img2img):
                return scripts.AlwaysVisible

            def ui(self, is_img2img):
                with gr.Accordion(
                    "⚡ ReActor V4 — RestoreFormer++ / CodeFormer + E4S Texture",
                    open=False,
                    elem_id="reactor_v4_accordion"
                ):
                    enabled = gr.Checkbox(
                        label="Enable ReActor V4",
                        value=False,
                        elem_id="reactor_v4_enabled"
                    )

                    gr.Markdown("""
**ReactorV4 Pipeline** *(ReactorV3 untouched)*
`inswapper` → **RestoreFormer++ / CodeFormer / GFPGAN** (ONNX, auto-download) → **E4S Texture Transfer**

All restoration models are safe ONNX from [FaceFusion HuggingFace](https://huggingface.co/facefusion/models-3.0.0) — no .pth, no pickle.
                    """)

                    # ── Source Image ──────────────────────────────────────────
                    with gr.Group():
                        gr.Markdown("### 📸 Source Face(s)")
                        with gr.Row():
                            source_image = gr.Image(
                                label="Source Face 1 (Main)",
                                type="pil",
                                interactive=True,
                                elem_id="reactor_v4_source"
                            )
                            source_image_2 = gr.Image(
                                label="Source Face 2 (Optional - for dual swapping)",
                                type="pil",
                                interactive=True,
                                elem_id="reactor_v4_source_2"
                            )

                    # ── Swapper ───────────────────────────────────────────────
                    with gr.Group():
                        gr.Markdown("### 🔀 Face Swapper")
                        with gr.Row():
                            swapper_model = gr.Dropdown(
                                label="Swapper Model",
                                choices=_swapper_choices(),
                                value="inswapper_128.onnx",
                                elem_id="reactor_v4_swapper"
                            )
                            refresh_btn = gr.Button("🔄", variant="secondary", size="sm", scale=0)
                        with gr.Row():
                            source_idx = gr.Slider(0, 10, step=1, value=0,
                                label="Source Face 1 Index", elem_id="reactor_v4_src_idx")
                            target_idx = gr.Slider(0, 10, step=1, value=0,
                                label="Target Face 1 Index", elem_id="reactor_v4_tgt_idx")
                        with gr.Row():
                            source_idx_2 = gr.Slider(0, 10, step=1, value=0,
                                label="Source Face 2 Index", elem_id="reactor_v4_src_idx_2")
                            target_idx_2 = gr.Slider(0, 10, step=1, value=1,
                                label="Target Face 2 Index", elem_id="reactor_v4_tgt_idx_2")
                        with gr.Row():
                            gender_match = gr.Dropdown(
                                label="Gender Matching",
                                choices=[
                                    ("Smart — same gender only", "S"),
                                    ("All — ignore gender",      "A"),
                                    ("Male faces only",          "M"),
                                    ("Female faces only",        "F"),
                                ],
                                value="S",
                                elem_id="reactor_v4_gender"
                            )
                            swap_strength = gr.Slider(
                                0.1, 1.0, step=0.05, value=1.0,
                                label="Swap Strength",
                                elem_id="reactor_v4_swap_str"
                            )

                    # ── Restoration ───────────────────────────────────────────
                    with gr.Group():
                        gr.Markdown("### 🔬 Face Restoration (Safe ONNX)")
                        with gr.Row():
                            enable_restore = gr.Checkbox(
                                label="Enable Restoration",
                                value=True,
                                elem_id="reactor_v4_restore_en"
                            )
                            restorer_strength = gr.Slider(
                                0.1, 1.0, step=0.05, value=1.0,
                                label="Restoration Strength",
                                info="1.0 = full, 0.5 = blend 50/50 with swapped",
                                elem_id="reactor_v4_restore_str"
                            )

                        restorer_model = gr.Dropdown(
                            label="Restoration Model",
                            choices=_restorer_choices(),
                            value="restoreformer_plus_plus",
                            elem_id="reactor_v4_restorer",
                            info="📥 = will auto-download on first use  ✅ = already on disk"
                        )
                        restorer_status = gr.Markdown(
                            _restorer_status("restoreformer_plus_plus"),
                            elem_id="reactor_v4_restore_status"
                        )

                        gr.Markdown("""
| Model | Quality | Skin | Size |
|---|---|---|---|
| **restoreformer_plus_plus** 🥇 | Most natural | Real skin texture, transformer | ~306 MB |
| **codeformer** 🥈 | Best overall | Identity-safe, fidelity control | ~377 MB |
| **gfpgan_1.4** | Reliable | GAN, balanced | ~340 MB |
| **gpen_bfr_512** | Sharp | Already in V3 dir | ~75 MB |
| **gpen_bfr_1024** | High-res | More detail | ~285 MB |
| **gpen_bfr_2048** | Max sharp | Can over-process | ~286 MB |

*All auto-downloaded from [FaceFusion HuggingFace](https://huggingface.co/facefusion/models-3.0.0) — ONNX, no pickle.*
                        """)

                    # ── Texture Transfer ──────────────────────────────────────
                    with gr.Group():
                        gr.Markdown("### 🎨 E4S Reference-Guided Texture")
                        enable_texture = gr.Checkbox(
                            label="Enable E4S Texture Transfer",
                            value=True,
                            elem_id="reactor_v4_tex_en"
                        )
                        with gr.Row():
                            texture_strength = gr.Slider(
                                0.0, 1.0, step=0.05, value=0.45,
                                label="Texture Strength",
                                info="How much reference skin colour/tone to inject",
                                elem_id="reactor_v4_tex_str"
                            )
                            hf_strength = gr.Slider(
                                0.0, 1.0, step=0.05, value=0.35,
                                label="HF Skin Injection",
                                info="High-frequency skin pore/texture from reference",
                                elem_id="reactor_v4_hf_str"
                            )

                    # ── Advanced ──────────────────────────────────────────────
                    with gr.Accordion("⚙️ Advanced", open=False):
                        with gr.Row():
                            occlusion_en = gr.Checkbox(
                                label="Occlusion Handling (hair/glasses)",
                                value=True, elem_id="reactor_v4_occ_en"
                            )
                            detail_enhance_en = gr.Checkbox(
                                label="👁️ Reference Detail Enhancement (Eyes + Teeth)",
                                value=True, elem_id="reactor_v4_detail_en",
                                info="Adaptive — auto-detects degradation and injects source detail"
                            )
                        with gr.Row():
                            aggressive_cleanup = gr.Checkbox(
                                label="Aggressive VRAM Cleanup",
                                value=False, elem_id="reactor_v4_aggressive"
                            )
                        occlusion_str = gr.Slider(
                            0.0, 1.0, step=0.1, value=1.0,
                            label="Occlusion Strength", elem_id="reactor_v4_occ_str"
                        )

                    # ── Pipeline diagram ──────────────────────────────────────
                    with gr.Accordion("ℹ️ Pipeline Info", open=False):
                        gr.Markdown("""
```
Source Image ──────────────────────────────────────┐
Target Image → Face Detect → inswapper              │
                    ↓                               │
            Occlusion Preserve                      │
                    ↓                               │
    Safe ONNX Restorer (auto-download)              │
    RestoreFormer++ / CodeFormer / GFPGAN           │
                    ↓                               │
    👁️ Reference Detail Enhancement ◄───────────────┤
    ■ Adaptive eye detail from source               │
    ■ Adaptive teeth detail from source             │
    ■ Eye colour preservation (Lab a/b)             │
                    ↓                               │
    E4S Texture Transfer ◄──────────────────────────┘
    ■ Lab histogram colour match (skin regions)
    ■ HF skin texture from reference image
    ■ Plasticity-adaptive blend
                    ↓
               OUTPUT
```
**ReactorV3 is completely untouched.**
                        """)

                # ── Events ────────────────────────────────────────────────────
                def on_refresh():
                    return (
                        gr.Dropdown.update(choices=_swapper_choices()),
                        gr.Dropdown.update(choices=_restorer_choices()),
                    )

                def on_restorer_change(key):
                    return _restorer_status(key)

                refresh_btn.click(
                    fn=on_refresh, inputs=[],
                    outputs=[swapper_model, restorer_model]
                )
                restorer_model.change(
                    fn=on_restorer_change, inputs=[restorer_model],
                    outputs=[restorer_status]
                )

                return [
                    enabled, source_image, source_image_2,
                    swapper_model, source_idx, target_idx, source_idx_2, target_idx_2,
                    gender_match, swap_strength,
                    enable_restore, restorer_model, restorer_strength,
                    enable_texture, texture_strength, hf_strength,
                    occlusion_en, occlusion_str,
                    aggressive_cleanup,
                    detail_enhance_en,
                ]

            def process(self, p,
                        enabled, source_image, source_image_2,
                        swapper_model, source_idx, target_idx, source_idx_2, target_idx_2,
                        gender_match, swap_strength,
                        enable_restore, restorer_model, restorer_strength,
                        enable_texture, texture_strength, hf_strength,
                        occlusion_en, occlusion_str,
                        aggressive_cleanup,
                        detail_enhance_en):
                if not enabled or source_image is None:
                    return
                # Ensure scripts dir is on path (Forge may call from fresh context)
                import sys as _sys
                _scripts_dir = os.path.dirname(os.path.abspath(__file__))
                if _scripts_dir not in _sys.path:
                    _sys.path.insert(0, _scripts_dir)
                p.rv4_enabled          = True
                p.rv4_source           = source_image
                p.rv4_source_2         = source_image_2
                p.rv4_swapper          = swapper_model
                p.rv4_src_idx          = int(source_idx)
                p.rv4_tgt_idx          = int(target_idx)
                p.rv4_src_idx_2        = int(source_idx_2)
                p.rv4_tgt_idx_2        = int(target_idx_2)
                p.rv4_gender           = gender_match
                p.rv4_swap_str         = float(swap_strength)
                p.rv4_restore_en       = bool(enable_restore)
                p.rv4_restorer         = restorer_model
                p.rv4_restore_str      = float(restorer_strength)
                p.rv4_texture_en       = bool(enable_texture)
                p.rv4_tex_str          = float(texture_strength)
                p.rv4_hf_str           = float(hf_strength)
                p.rv4_occ_en           = bool(occlusion_en)
                p.rv4_occ_str          = float(occlusion_str)
                p.rv4_aggressive       = bool(aggressive_cleanup)
                p.rv4_detail_en        = bool(detail_enhance_en)

            def postprocess_image(self, p, pp,
                                  enabled, source_image, source_image_2,
                                  swapper_model, source_idx, target_idx, source_idx_2, target_idx_2,
                                  gender_match, swap_strength,
                                  enable_restore, restorer_model, restorer_strength,
                                  enable_texture, texture_strength, hf_strength,
                                  occlusion_en, occlusion_str,
                                  aggressive_cleanup,
                                  detail_enhance_en):
                if not enabled or source_image is None:
                    return
                if not getattr(p, "rv4_enabled", False):
                    return
                # Respect Forge Interrupt button
                try:
                    from modules import shared as _sh
                    if getattr(_sh.state, "interrupted", False) or getattr(_sh.state, "skipped", False):
                        print(f"{TAG} ✋ Skipped — user pressed Interrupt")
                        return
                except ImportError:
                    pass
                try:
                    # Re-inject scripts dir — Forge may call this in a fresh
                    # context where the outer sys.path insertion is not visible.
                    import sys as _sys
                    _scripts_dir = os.path.dirname(os.path.abspath(__file__))
                    if _scripts_dir not in _sys.path:
                        _sys.path.insert(0, _scripts_dir)

                    from reactor_v4_pipeline import get_reactor_v4_pipeline
                    mp = os.path.join(_webui_root(), "models")
                    pipeline = get_reactor_v4_pipeline(mp)
                    pipeline.configure(
                        restorer_model     = p.rv4_restorer,
                        restorer_strength  = p.rv4_restore_str,
                        enable_restoration = p.rv4_restore_en,
                        enable_texture     = p.rv4_texture_en,
                        texture_strength   = p.rv4_tex_str,
                        hf_strength        = p.rv4_hf_str,
                        swap_strength      = p.rv4_swap_str,
                        gender_match       = p.rv4_gender,
                        occlusion_enabled  = p.rv4_occ_en,
                        occlusion_strength = p.rv4_occ_str,
                        enable_detail_enhance = p.rv4_detail_en,
                        auto_cleanup       = True,
                        aggressive_cleanup = p.rv4_aggressive,
                    )
                    if isinstance(p.rv4_source, Image.Image):
                        src = cv2.cvtColor(np.array(p.rv4_source), cv2.COLOR_RGB2BGR)
                    else:
                        src = p.rv4_source

                    src2 = None
                    if p.rv4_source_2 is not None:
                        if isinstance(p.rv4_source_2, Image.Image):
                            src2 = cv2.cvtColor(np.array(p.rv4_source_2), cv2.COLOR_RGB2BGR)
                        else:
                            src2 = p.rv4_source_2

                    tgt = cv2.cvtColor(np.array(pp.image), cv2.COLOR_RGB2BGR)
                    print(f"{TAG} ▸ UI source: type={type(p.rv4_source).__name__}, shape after cvt={src.shape}")
                    if src2 is not None:
                        print(f"{TAG} ▸ UI source 2: type={type(p.rv4_source_2).__name__}, shape after cvt={src2.shape}")
                    print(f"{TAG} ▸ UI target (pp.image): PIL size={pp.image.size}, shape after cvt={tgt.shape}")
                    result_bgr, msg = pipeline.process(
                        src, tgt, p.rv4_src_idx, p.rv4_tgt_idx, p.rv4_swapper,
                        source_img_2=src2,
                        source_face_idx_2=p.rv4_src_idx_2,
                        target_face_idx_2=p.rv4_tgt_idx_2,
                    )
                    # Diagnostic: compare original vs result
                    diff_check = cv2.absdiff(tgt, result_bgr)
                    mean_d = float(diff_check.mean())
                    print(f"{TAG} ▸ postprocess pixel diff (tgt vs result): mean={mean_d:.2f}")
                    if mean_d < 0.5:
                        print(f"{TAG} ⚠ Pipeline returned near-identical image!")

                    pp.image = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
                    print(f"{TAG} ▸ pp.image updated: size={pp.image.size}, mode={pp.image.mode}")
                    print(f"{TAG} {msg}")
                except Exception as exc:
                    import traceback; traceback.print_exc()
                    print(f"{TAG} ERROR: {exc}")

        return ReactorV4Script

    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"[ReactorV4] Script creation error: {exc}")

        class _Dummy:
            def title(self): return "ReActor V4 (Error)"
            def show(self, _): return False
            def ui(self, _): return []

        return _Dummy


Script = create_script()
