#!/usr/bin/env python3
"""
Pre-warm / Descarga de Modelos GGUF hacia Google Drive (5 TB)
Arquitectura Robusta en Dos Fases:
1. Descarga a disco local NVMe ultra-rápido (/content/hf_cache)
2. Sincronización síncrona hacia Google Drive vía `rclone copy` con verificación estricta de bytes
Garantiza que la VM no se apague antes de que el archivo esté 100% persistido en Drive.
"""
import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

# 1. Configurar directorio local efímero de alta velocidad
local_cache_dir = Path("/content/hf_cache")
local_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(local_cache_dir)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print(f"⚡ Directorio de descarga local rápida (NVMe): {local_cache_dir}")

# 2. Asegurar rclone.conf en /root/.config/rclone/
rclone_conf = Path("/root/.config/rclone/rclone.conf")
content_conf = Path("/content/rclone.conf")
if not rclone_conf.exists() and content_conf.exists():
    rclone_conf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(content_conf, rclone_conf)

if not Path("/usr/bin/rclone").exists() and not Path("/usr/local/bin/rclone").exists():
    print("📦 Instalando binario oficial de rclone...")
    subprocess.run("curl -s https://rclone.org/install.sh | bash", shell=True, check=True)

# 3. Instalar huggingface_hub si no está disponible
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("📦 Instalando huggingface_hub...")
    subprocess.run("pip install -q huggingface_hub", shell=True, check=True)
    from huggingface_hub import hf_hub_download

MODELS_TO_DOWNLOAD = [
    {
        "tier": "Tier 1 (T4)",
        "repo": "mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-i1-GGUF",
        "file": "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf"
    }
]

if len(sys.argv) > 1:
    arg = sys.argv[1].lower()
    if arg in ("1", "tier1", "t4"):
        MODELS_TO_DOWNLOAD = [{
            "tier": "Tier 1 (T4)",
            "repo": "mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-i1-GGUF",
            "file": "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf"
        }]
    elif arg in ("2", "tier2", "l4"):
        MODELS_TO_DOWNLOAD = [{
            "tier": "Tier 2 (L4)",
            "repo": "Abiray/Huihui-Qwen3.6-35B-A3B-abliterated-GGUF",
            "file": "Huihui-Qwen3.6-35B-A3B-abliterated-Q3_K_M.gguf"
        }]
    elif arg in ("3", "tier3", "kimi"):
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
    elif arg in ("3,4", "3-4", "heavy", "big", "grandes"):
        MODELS_TO_DOWNLOAD = [
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
    print("\n" + "=" * 75)
    print(f"📥 [Fase 1/2] Descargando {item['tier']} a disco local: {item['repo']} -> {item['file']}")
    print("=" * 75)
    t0 = time.time()
    try:
        path = hf_hub_download(
            repo_id=item["repo"],
            filename=item["file"],
            resume_download=True
        )
        t_elap = time.time() - t0
        size_gb = os.path.getsize(path) / (1024 ** 3)
        print(f"   ✅ Descarga local completada en {t_elap:.1f}s ({size_gb:.2f} GB a ~{size_gb / (t_elap / 60):.2f} GB/min)")
        print(f"   Ruta local: {path}")

        # Fase 2: Subida síncrona a Google Drive vía rclone copy
        print("\n" + "─" * 75)
        print(f"📤 [Fase 2/2] Sincronizando {item['file']} directamente a Google Drive (5 TB)...")
        print("─" * 75)
        t_sync_start = time.time()
        
        # Subir todo el árbol de cache local manteniendo la estructura de hub/
        sync_cmd = [
            "rclone", "copy",
            str(local_cache_dir / "hub"),
            "gdrive:ColabModelsCache/huggingface/hub",
            "--transfers", "4",
            "--checkers", "8",
            "--buffer-size", "128M",
            "--stats", "15s",
            "-v"
        ]
        subprocess.run(sync_cmd, check=True)
        t_sync = time.time() - t_sync_start
        print(f"   🎉 ¡Sincronización confirmada en Google Drive en {t_sync:.1f}s!")

    except Exception as e:
        print(f"❌ Error al procesar {item['file']}: {e}")

print("\n🏁 Proceso de pre-warm y sincronización a Drive finalizado al 100%.")
