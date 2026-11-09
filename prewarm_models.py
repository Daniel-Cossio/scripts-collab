#!/usr/bin/env python3
"""
Pre-warm / Descarga de Modelos GGUF hacia Google Drive (5 TB)
Ejecutable en Colab VM (CPU o GPU) para poblar ColabModelsCache/huggingface
sin re-descargas posteriores.
"""
import os
import sys
import time
from pathlib import Path

# 1. Asegurar montaje de Drive
import setup_drive_headless
if not setup_drive_headless.ensure_drive_mounted():
    print("❌ Error: No se pudo montar Google Drive.")
    sys.exit(1)

my_drive = Path("/content/drive/MyDrive")
hf_cache_dir = my_drive / "ColabModelsCache" / "huggingface"
hf_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(hf_cache_dir)

print(f"💾 Directorio de caché persistente: {hf_cache_dir}")

# 2. Instalar huggingface_hub si no está disponible
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("📦 Instalando huggingface_hub...")
    import subprocess
    subprocess.run("pip install -q huggingface_hub", shell=True, check=True)
    from huggingface_hub import hf_hub_download

MODELS_TO_DOWNLOAD = [
    {
        "tier": "Tier 1 (T4)",
        "repo": "mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-i1-GGUF",
        "file": "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf"
    }
]

# Permitir pasar argumento de qué modelo descargar
if len(sys.argv) > 1:
    arg = sys.argv[1].lower()
    if arg in ("3", "tier3", "kimi"):
        MODELS_TO_DOWNLOAD = [{
            "tier": "Tier 3 (A100 40GB)",
            "repo": "mradermacher/Kimi-Linear-48B-A3B-Instruct-abliterated-bf16-i1-GGUF",
            "file": "Kimi-Linear-48B-A3B-Instruct-abliterated-bf16.i1-Q4_K_M.gguf"
        }]
    elif arg in ("4", "tier4", "apex"):
        MODELS_TO_DOWNLOAD = [{
            "tier": "Tier 4 (A100 80GB)",
            "repo": "mudler/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-APEX-GGUF",
            "file": "Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-APEX-Quality.gguf"
        }]
    elif arg in ("all", "todos"):
        MODELS_TO_DOWNLOAD = [
            {
                "tier": "Tier 1 (T4)",
                "repo": "mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-i1-GGUF",
                "file": "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf"
            },
            {
                "tier": "Tier 3 (A100 40GB)",
                "repo": "mradermacher/Kimi-Linear-48B-A3B-Instruct-abliterated-bf16-i1-GGUF",
                "file": "Kimi-Linear-48B-A3B-Instruct-abliterated-bf16.i1-Q4_K_M.gguf"
            },
            {
                "tier": "Tier 4 (A100 80GB)",
                "repo": "mudler/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-APEX-GGUF",
                "file": "Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-APEX-Quality.gguf"
            }
        ]

for item in MODELS_TO_DOWNLOAD:
    print("\n" + "=" * 70)
    print(f"📥 Descargando {item['tier']}: {item['repo']} -> {item['file']}")
    print("=" * 70)
    t0 = time.time()
    try:
        path = hf_hub_download(
            repo_id=item["repo"],
            filename=item["file"],
            local_dir_use_symlinks=False,
            resume_download=True
        )
        t_elap = time.time() - t0
        size_gb = os.path.getsize(path) / (1024 ** 3)
        print(f"✅ ¡Descarga completada con éxito!")
        print(f"   Ruta: {path}")
        print(f"   Tamaño: {size_gb:.2f} GB")
        print(f"   Tiempo transcurrido: {t_elap:.1f}s (~{size_gb / (t_elap / 60):.2f} GB/min)")
    except Exception as e:
        print(f"❌ Error al descargar {item['file']}: {e}")

print("\n🏁 Proceso de pre-warm finalizado.")
