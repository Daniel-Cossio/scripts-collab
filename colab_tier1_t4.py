# ==============================================================================
# SERVIDOR LLAMA-SERVER EN GOOGLE COLAB - TIER 1: TESLA T4 (15.3 GB VRAM)
# Especialidad: Agilidad, Contexto Masivo (128k Tokens) y Cero Costo de Créditos
# ==============================================================================
import subprocess
import time
import re
import os
import sys
import glob
from pathlib import Path

# ------------------------------------------------------------------------------
# 0. Selector de Modelos Tier 1 (Optimizados para 15 GB VRAM)
# ------------------------------------------------------------------------------
# Selecciona cambiando este índice ("1", "2" o "3"):
MODELO_ELEGIDO = os.environ.get("COLAB_MODEL_TIER1", "1")

CATALOGO_MODELOS = {
    "1": {
        "repo": "mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-i1-GGUF",
        "file": "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf",
        "alias": "deepseek-v4-qwen3.5:9b",
        "desc": "🧠 DeepSeek-V4-Pro Qwen 3.5 9B Uncensored (Heretic ARA / 128k ctx - 45 tok/s)",
        "context": "131072"
    },
    "2": {
        "repo": "RavichandranJ/Dolphin3-Cyber-8B-GGUF:Q4_K_M",
        "file": None,
        "alias": "dolphin3-cyber:8b",
        "desc": "🛡️ Dolphin 3.0 Cyber 8B (SecOps, Red Team, Exploits & CTFs)",
        "context": "65536"
    },
    "3": {
        "repo": "Qwen/Qwen3-8B-GGUF:Q4_K_M",
        "file": None,
        "alias": "qwen3:8b",
        "desc": "⚡ Qwen3 8B Instruct (Razonamiento Dinámico /think - 128k ctx)",
        "context": "131072"
    }
}

CFG_MODELO = CATALOGO_MODELOS.get(MODELO_ELEGIDO, CATALOGO_MODELOS["1"])
REPO_MODELO = CFG_MODELO["repo"]
ARCHIVO_MODELO = CFG_MODELO["file"]
ALIAS_MODELO = CFG_MODELO["alias"]
DESCRIPCION = CFG_MODELO["desc"]
CONTEXT_TOKENS = CFG_MODELO["context"]

print("=" * 75)
print("🚀 GOOGLE COLAB LLAMA-SERVER | TIER 1: TESLA T4")
print(f"🎯 Modelo activo:      {DESCRIPCION}")
print(f"📦 Repositorio GGUF:   {REPO_MODELO}")
if ARCHIVO_MODELO:
    print(f"📄 Archivo específico: {ARCHIVO_MODELO}")
print(f"📏 Ventana contexto:   {int(CONTEXT_TOKENS):,} tokens")
print("=" * 75)

# ------------------------------------------------------------------------------
# 1. Verificación de GPU NVIDIA
# ------------------------------------------------------------------------------
print("🔍 1/4 Verificando GPU NVIDIA...")
try:
    gpu_info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True
    ).strip()
    print(f"🎮 GPU Detectada: {gpu_info}")
except Exception:
    print("\n" + "!" * 75)
    print("❌ ERROR: No se detectó ninguna GPU NVIDIA activa.")
    print("En Colab ve a: Entorno de ejecución -> Cambiar tipo de entorno -> T4 GPU")
    print("!" * 75 + "\n")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 2. Limpieza de Procesos Previos
# ------------------------------------------------------------------------------
print("🧹 Limpiando procesos previos y liberando VRAM...")
subprocess.run("pkill -9 -f llama-server || true", shell=True)
subprocess.run("pkill -9 cloudflared || true", shell=True)
time.sleep(1)

# ------------------------------------------------------------------------------
# 3. Descargar Túnel Cloudflared
# ------------------------------------------------------------------------------
if not Path("/usr/local/bin/cloudflared").exists():
    print("📦 2/4 Descargando cliente de túnel Cloudflare...")
    subprocess.run(
        "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared",
        shell=True,
        check=True
    )

# ------------------------------------------------------------------------------
# 4. Descargar Binarios Precompilados Oficiales de llama-server + CUDA Runtime
# ------------------------------------------------------------------------------
print("📦 3/4 Verificando motor de inferencia llama.cpp con CUDA 12.8...")
LLAMA_TAG = "b12015"
BIN_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/llama-{LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz"
CUDA_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/cudart-llama-{LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz"

subprocess.run("mkdir -p /opt/llama", shell=True, check=True)

if not Path("/opt/llama/llama-server").exists():
    print("   -> Descargando binarios llama-server...")
    subprocess.run(f"curl -sL {BIN_URL} | tar -xzf - -C /opt/llama --strip-components=1", shell=True, check=True)
    subprocess.run("chmod +x /opt/llama/llama-server", shell=True, check=True)

if not Path("/opt/llama/libcublas.so.12").exists():
    print("   -> Descargando runtime CUDA cuBLAS...")
    subprocess.run(f"curl -sL {CUDA_URL} | tar -xzf - -C /opt/llama --strip-components=1", shell=True, check=True)

# ------------------------------------------------------------------------------
# 5. Configurar Enlace Dinámico de Librerías CUDA
# ------------------------------------------------------------------------------
compat_dirs = glob.glob("/usr/local/cuda*/compat")
nvidia_dirs = ["/opt/llama", "/usr/lib64-nvidia", "/usr/local/cuda/lib64"] + compat_dirs
os.environ["LD_LIBRARY_PATH"] = ":".join(nvidia_dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["PATH"] = f"/opt/llama:{os.environ.get('PATH', '')}"

# ------------------------------------------------------------------------------
# 6. Lanzar llama-server con Aceleración GPU Real (Puerto 8899)
# ------------------------------------------------------------------------------
print(f"\n⚙️  4/4 Iniciando llama-server ({ALIAS_MODELO}) con {CONTEXT_TOKENS} tokens...")
server_cmd = [
    "/opt/llama/llama-server",
    "-hf", REPO_MODELO,
    "--alias", ALIAS_MODELO,
    "--host", "0.0.0.0",
    "--port", "8899",
    "-ngl", "99",
    "-c", str(CONTEXT_TOKENS),
    "-ctk", "q8_0",
    "-ctv", "q8_0",
    "--flash-attn", "on",
    "--no-mmproj",
    "--cont-batching",
    "--jinja"
]

if ARCHIVO_MODELO:
    server_cmd.extend(["-hff", ARCHIVO_MODELO])

server_proc = subprocess.Popen(
    server_cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

ready = False
for line in server_proc.stdout:
    line_clean = line.strip()
    if any(k in line for k in ["ggml_cuda", "offloaded", "HTTP server listening", "listening on", "load:"]):
        print(f"   🚀 {line_clean}", flush=True)
    elif "error" in line.lower() or "invalid" in line.lower():
        print(f"   ⚠️ {line_clean}", flush=True)

    if "HTTP server listening" in line or "listening on" in line:
        ready = True
        break

if not ready or server_proc.poll() is not None:
    print("\n❌ Error crítico: llama-server no pudo iniciar.")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 7. Apertura de Túnel Cloudflare HTTPS
# ------------------------------------------------------------------------------
print("\n🌐 Abriendo túnel seguro Cloudflare hacia tu entorno local...")
tunnel = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://localhost:8899", "--http-host-header", "localhost:8899"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

url_encontrada = None
for line in tunnel.stdout:
    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
    if match:
        url_encontrada = match.group(0)
        break

if not url_encontrada:
    print("❌ No se pudo obtener la URL de Cloudflare.")
else:
    print("\n" + "=" * 75)
    print("✅ ¡SERVIDOR LISTO CON GPU TESLA T4!")
    print(f"🧠 Modelo:        {ALIAS_MODELO}")
    print(f"📏 Contexto:      {int(CONTEXT_TOKENS):,} tokens (Flash Attention + Q8 KV Cache)")
    print(f"🔗 URL del Túnel: {url_encontrada}")
    print("=" * 75)
    print("\n👉 EN LA TERMINAL DE TU WORKSTATION / LAPTOP, EJECUTA:\n")
    print(f"   colab-connect {url_encontrada}")
    print("\n" + "=" * 75)

try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("\n🛑 Servidor detenido.")
