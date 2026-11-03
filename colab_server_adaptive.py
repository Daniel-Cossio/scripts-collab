# ==============================================================================
# SERVIDOR LLAMA-SERVER ADAPTATIVO EN GOOGLE COLAB (MULTI-TIER AUTO-DETECT)
# Detecta automáticamente la GPU asignada (Tesla T4, NVIDIA L4, NVIDIA A100)
# y optimiza el modelo, la VRAM y la ventana de contexto.
# ==============================================================================
import subprocess
import time
import re
import os
import sys
import glob
from pathlib import Path

# ------------------------------------------------------------------------------
# 1. Detección Inteligente de GPU NVIDIA en Colab
# ------------------------------------------------------------------------------
print("=" * 75)
print("🔍 1/5 Detectando GPU NVIDIA y evaluando capacidad de cómputo...")
try:
    gpu_raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        text=True
    ).strip()
    gpu_name, gpu_vram_str = [x.strip() for x in gpu_raw.split(",")]
    gpu_vram_mb = int(gpu_vram_str)
    print(f"🎮 GPU Detectada: {gpu_name} ({gpu_vram_mb:,} MB VRAM)")
except Exception as e:
    print("\n" + "!" * 75)
    print("❌ ERROR: No se detectó ninguna GPU NVIDIA activa.")
    print("En Colab ve a: Entorno de ejecución -> Cambiar tipo de entorno de ejecución.")
    print("Elige T4 GPU (Gratis), L4 GPU (Pro) o A100 GPU (Pro+).")
    print("!" * 75 + "\n")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 2. Asignación de Tier según Capacidad de Hardware
# ------------------------------------------------------------------------------
# Puedes forzar un modelo específico mediante la variable de entorno COLAB_MODEL
MODEL_OVERRIDE = os.environ.get("COLAB_MODEL", None)

if gpu_vram_mb <= 16500 or "t4" in gpu_name.lower():
    TIER_ID = "TIER_1_T4"
    TIER_LABEL = "Tier 1: Tesla T4 (Costo Cero / Agilidad)"
    TIER_CATALOG = {
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
    DEFAULT_KEY = "1"

elif gpu_vram_mb <= 28000 or "l4" in gpu_name.lower():
    TIER_ID = "TIER_2_L4"
    TIER_LABEL = "Tier 2: NVIDIA L4 24GB (SWE Agéntico Profesional)"
    TIER_CATALOG = {
        "1": {
            "repo": "Abiray/Huihui-Qwen3.6-35B-A3B-abliterated-GGUF",
            "file": "Huihui-Qwen3.6-35B-A3B-abliterated-Q3_K_M.gguf",
            "alias": "qwen3.6-35b-a3b:uncensored",
            "desc": "⚡ Huihui Qwen 3.6 35B-A3B MoE (Abliterated / Cero Filtros / 128k ctx - ~70 tok/s)",
            "context": "131072"
        },
        "2": {
            "repo": "mradermacher/Qwen3.8-35B-A3B-Distill-i1-GGUF",
            "file": "Qwen3.8-35B-A3B-Distill.i1-IQ4_XS.gguf",
            "alias": "qwen3.8-35b-a3b:moe",
            "desc": "⚡ Qwen 3.8 35B-A3B MoE Distill (2026 / Alta Velocidad ~70 tok/s - 32k ctx)",
            "context": "32768"
        },
        "3": {
            "repo": "Jackrong/Qwopus3.8-27B-Coder-Flash-GGUF:Q4_K_M",
            "file": None,
            "alias": "qwopus3.8-coder:27b",
            "desc": "💻 Qwopus 3.8 27B Coder Flash (Qwen + Claude Opus Merge - 64k ctx)",
            "context": "65536"
        },
        "4": {
            "repo": "0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF:Q4_K_M",
            "file": None,
            "alias": "qwen3.8-uncensored-rvn:27b",
            "desc": "🛡️ Qwen 3.8 27B Heretic RVN (Doble Pasada Abliterated / Cero Filtros - 64k ctx)",
            "context": "65536"
        }
    }
    DEFAULT_KEY = "1"

else:
    TIER_ID = "TIER_3_A100"
    TIER_LABEL = f"Tier 3: NVIDIA A100 ({gpu_vram_mb // 1024}GB VRAM / Frontier Level)"
    TIER_CATALOG = {
        "1": {
            "repo": "bartowski/huihui-ai_DeepSeek-R1-Distill-Llama-70B-abliterated-GGUF:Q4_K_M",
            "file": None,
            "alias": "deepseek-r1-llama:70b",
            "desc": "🧠 DeepSeek-R1 Distill Llama 70B Abliterated (Máximo Razonamiento Frontier / Sin Filtros)",
            "context": "65536" if gpu_vram_mb < 60000 else "131072"
        },
        "2": {
            "repo": "bartowski/Llama-3.3-70B-Instruct-abliterated-GGUF:Q4_K_M",
            "file": None,
            "alias": "llama-3.3-abliterated:70b",
            "desc": "🏗️ Llama 3.3 70B Instruct Abliterated (Estándar Oro SWE & Refactorización Masiva)",
            "context": "65536"
        },
        "3": {
            "repo": "bartowski/huihui-ai_DeepSeek-R1-Distill-Llama-70B-abliterated-GGUF:Q3_K_M",
            "file": None,
            "alias": "deepseek-r1-llama:70b-q3",
            "desc": "⚡ DeepSeek-R1 Distill Llama 70B Q3_K_M (A100 40GB Optimizado / 64k ctx)",
            "context": "65536"
        }
    }
    DEFAULT_KEY = "1"

SELECCION = MODEL_OVERRIDE if MODEL_OVERRIDE in TIER_CATALOG else DEFAULT_KEY
CFG = TIER_CATALOG[SELECCION]
REPO_MODELO = CFG["repo"]
ARCHIVO_MODELO = CFG["file"]
ALIAS_MODELO = CFG["alias"]
DESCRIPCION = CFG["desc"]
CONTEXT_TOKENS = CFG["context"]

print(f"🎯 Nivel Asignado:     {TIER_LABEL}")
print(f"🧠 Modelo Seleccionado: {DESCRIPCION}")
print(f"📦 Repositorio GGUF:   {REPO_MODELO}")
if ARCHIVO_MODELO:
    print(f"📄 Archivo específico: {ARCHIVO_MODELO}")
print(f"📏 Ventana de Contexto: {int(CONTEXT_TOKENS):,} tokens (Flash Attention + Q8 KV Cache)")
print("=" * 75)

# ------------------------------------------------------------------------------
# 3. Limpieza de Procesos Previos
# ------------------------------------------------------------------------------
print("🧹 2/5 Limpiando procesos previos y liberando memoria VRAM...")
subprocess.run("pkill -9 -f llama-server || true", shell=True)
subprocess.run("pkill -9 cloudflared || true", shell=True)
time.sleep(1)

# ------------------------------------------------------------------------------
# 4. Descargar Túnel Cloudflared
# ------------------------------------------------------------------------------
if not Path("/usr/local/bin/cloudflared").exists():
    print("📦 3/5 Descargando cliente de túnel Cloudflare...")
    subprocess.run(
        "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared",
        shell=True,
        check=True
    )

# ------------------------------------------------------------------------------
# 5. Descargar Binarios Precompilados Oficiales de llama-server + CUDA Runtime
# ------------------------------------------------------------------------------
print("📦 4/5 Verificando motor de inferencia llama.cpp con CUDA 12.8...")
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

# Configurar Enlace Dinámico de Librerías CUDA
compat_dirs = glob.glob("/usr/local/cuda*/compat")
nvidia_dirs = ["/opt/llama", "/usr/lib64-nvidia", "/usr/local/cuda/lib64"] + compat_dirs
os.environ["LD_LIBRARY_PATH"] = ":".join(nvidia_dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["PATH"] = f"/opt/llama:{os.environ.get('PATH', '')}"

# ------------------------------------------------------------------------------
# 6. Lanzar llama-server con Aceleración GPU Real (Puerto 8899)
# ------------------------------------------------------------------------------
print(f"\n⚙️  5/5 Iniciando llama-server ({ALIAS_MODELO}) con {CONTEXT_TOKENS} tokens...")
server_cmd = [
    "/opt/llama/llama-server",
    "-hf", REPO_MODELO,
    "--alias", ALIAS_MODELO,
    "--host", "0.0.0.0",
    "--port", "8899",
    "-ngl", "99",
    "-c", str(CONTEXT_TOKENS),
    "-ctk", "q4_0" if int(CONTEXT_TOKENS) > 65536 else "q8_0",
    "-ctv", "q4_0" if int(CONTEXT_TOKENS) > 65536 else "q8_0",
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
    print("✅ ¡SERVIDOR ADAPTATIVO LISTO CON ACELERACIÓN GPU TOTAL!")
    print(f"🎮 GPU Detectada: {gpu_name} ({gpu_vram_mb:,} MB)")
    print(f"🧠 Modelo Activo: {ALIAS_MODELO}")
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
