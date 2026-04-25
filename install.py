"""
ReactorV4 install.py — dependency checker & installer.
Runs automatically when SD WebUI Forge loads the extension.
"""

import importlib
import subprocess
import sys
import os


def check_and_install(package: str, import_name: str = None, version: str = None):
    name = import_name or package
    try:
        importlib.import_module(name)
    except ImportError:
        pkg = f"{package}=={version}" if version else package
        print(f"[ReactorV4] Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])


# Required packages
check_and_install("insightface")
check_and_install("onnxruntime", import_name="onnxruntime")
check_and_install("facexlib")
check_and_install("gfpgan")
check_and_install("torchvision")

# OSDFace dependencies (diffusion pipeline)
check_and_install("diffusers")
check_and_install("transformers")
check_and_install("accelerate")

# Texture transfer VGG features
check_and_install("scipy")

print("[ReactorV4] All dependencies satisfied.")
print("")
print("[ReactorV4] ════════════════════════════════════════")
print("[ReactorV4]  OSDFace model download instructions:")
print("[ReactorV4]  1. Download weights from:")
print("[ReactorV4]     https://drive.google.com/drive/folders/1Nci6KufB8t2Uj-6tobrw3S7kkfQUTLHV")
print("[ReactorV4]  2. Place files in:")
print(f"[ReactorV4]     <webui>/models/reactor_v4/OSDFace/")
print("[ReactorV4]  OSDFace also needs Stable Diffusion 2.1 base model.")
print("[ReactorV4]  Download from HuggingFace: stabilityai/stable-diffusion-2-1-base")
print("[ReactorV4] ════════════════════════════════════════")
