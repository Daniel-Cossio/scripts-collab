# ==============================================================================
# SERVIDOR LLAMA-SERVER EN GOOGLE COLAB - TIER 3: NVIDIA A100 (40GB / 80GB VRAM)
# Especialidad: Nivel Frontier, Razonamiento Matemático/Arquitectónico Extremo
# ==============================================================================
import subprocess
import time
import re
import os
import sys
import glob
from pathlib import Path

# ------------------------------------------------------------------------------
# 0. Selector de Modelos Tier 3 (Optimizados para 40GB y 80GB VRAM)
# ------------------------------------------------------------------------------
MODELO_ELEGIDO = os.environ.get("COLAB_MODEL_TIER3", "1")

CATALOGO_MODELOS = {
    "1": {
        "repo": "mradermacher/Kimi-Linear-48B-A3B-Instruct-abliterated-bf16-i1-GGUF",
        "file": "Kimi-Linear-48B-A3B-Instruct-abliterated-bf16.i1-Q4_K_M.gguf",
        "alias": "kimi-linear-48b-a3b:uncensored",
        "desc": "🌙 Kimi Linear 48B-A3B Abliterated (31 Oct 2026 / MoE 3B activos + atención lineal / 512k ctx)",
        "context": "524288"
    },
    "2": {
        "repo": "mudler/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-APEX-GGUF",
        "file": "Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-APEX-Quality.gguf",
        "alias": "qwen3.6-opus:apex",
        "desc": "⚡ Huihui Qwen 3.6 35B-A3B Claude 4.7 Opus APEX (1 Nov 2026 / MoE Q6_K / 128k ctx)",
        "context": "131072"
    },
    "3": {
        "repo": "bartowski/Llama-3.3-70B-Instruct-abliterated-GGUF:Q3_K_M",
        "file": None,
        "alias": "llama-3.3-70b:uncensored",
        "desc": "🏗️ Llama 3.3 70B Instruct Abliterated (Denso / Q3_K_M / 64k ctx - ~25 tok/s)",
        "context": "65536"
    },
    "4": {
        "repo": "bartowski/huihui-ai_DeepSeek-R1-Distill-Llama-70B-abliterated-GGUF:Q3_K_M",
        "file": None,
        "alias": "deepseek-r1-70b:uncensored",
        "desc": "🧠 DeepSeek-R1 Distill Llama 70B Abliterated (Denso / Q3_K_M / 64k ctx)",
        "context": "65536"
    }
}

CFG_MODELO = CATALOGO_MODELOS.get(MODELO_ELEGIDO, CATALOGO_MODELOS["1"])
REPO_MODELO = CFG_MODELO["repo"]
ARCHIVO_MODELO = CFG_MODELO["file"]
ALIAS_MODELO = CFG_MODELO["alias"]
DESCRIPCION = CFG_MODELO["desc"]
CONTEXT_TOKENS = CFG_MODELO["context"]

print("=" * 75)
print("🚀 GOOGLE COLAB LLAMA-SERVER | TIER 3: NVIDIA A100 (40GB / 80GB)")
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
    print("En Colab ve a: Entorno de ejecución -> Cambiar tipo de entorno -> A100 GPU")
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

t_inicio = time.time()
t_descarga_fin = None
t_carga_tensores = None
ready = False

server_proc = subprocess.Popen(
    server_cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

for line in server_proc.stdout:
    line_clean = line.strip()

    # 1. Detección de progreso de descarga desde Hugging Face
    if any(k in line for k in ["downloading from", "common_download_file", "downloading"]):
        print(f"   📥 {line_clean}", flush=True)

    # 2. Detección de inicio de carga en memoria (fin de descarga o hit en caché)
    if any(k in line for k in ["llama_model_loader", "loaded meta data", "llm_load_vocab"]) and t_descarga_fin is None:
        t_descarga_fin = time.time()
        seg_descarga = t_descarga_fin - t_inicio
        if seg_descarga < 3.0:
            print(f"   📦 Modelo verificado en caché local ({seg_descarga:.2f}s - sin re-descarga)", flush=True)
        else:
            print(f"   ✅ Descarga completada desde Hugging Face en {seg_descarga:.1f}s", flush=True)

    # 3. Detección de carga de capas en GPU, buffers y KV Cache
    if any(k in line for k in ["ggml_cuda", "offloaded", "load_tensors", "compute buffer", "kv self size"]):
        print(f"   🚀 {line_clean}", flush=True)

    # 4. Extracción de métrica interna de carga de llama.cpp (ms)
    match_load = re.search(r"load time\s*=\s*([\d\.]+)\s*ms", line)
    if match_load:
        t_carga_tensores = float(match_load.group(1)) / 1000.0
        print(f"   ⚡ Carga pura de tensores (llama.cpp): {t_carga_tensores:.2f}s", flush=True)

    # 5. Detección de errores
    if "error" in line.lower() or "invalid" in line.lower():
        print(f"   ⚠️ {line_clean}", flush=True)

    # 6. Detección de servidor activo
    if "HTTP server listening" in line or "listening on" in line:
        ready = True
        break

t_fin = time.time()
tiempo_total = t_fin - t_inicio

if not ready or server_proc.poll() is not None:
    print("\n❌ Error crítico: llama-server no pudo iniciar.")
    sys.exit(1)

# Reporte estructurado de métricas de tiempo
t_vram = (t_fin - t_descarga_fin) if t_descarga_fin else tiempo_total
seg_descarga = (t_descarga_fin - t_inicio) if t_descarga_fin else 0.0

print("\n" + "─" * 75)
print("⏱️  MÉTRICAS DE TIEMPO DE CARGA:")
if seg_descarga < 3.0:
    print(f"   • 📦 Verificación en Caché:     {seg_descarga:.2f} s (Hit en caché)")
else:
    print(f"   • 📥 Descarga Hugging Face:     {seg_descarga:.1f} s")
print(f"   • 🧠 Carga a VRAM y KV Cache:   {t_vram:.1f} s")
if t_carga_tensores:
    print(f"   • ⚡ Lectura de Tensores GPU:   {t_carga_tensores:.2f} s")
print(f"   • 🏁 Tiempo Total Preparación:  {tiempo_total:.1f} s")
print("─" * 75)

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
    print("✅ ¡SERVIDOR LISTO CON GPU NVIDIA A100!")
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
