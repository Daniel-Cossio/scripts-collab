# ==============================================================================
# SERVIDOR LLAMA-SERVER EN GOOGLE COLAB - TIER 4: NVIDIA A100 (80 GB VRAM)
# Especialidad: Modelos Frontier 70B/72B en Q4_K_M, 128k Contexto y Q8 KV Cache
# ==============================================================================
import subprocess
import time
import re
import os
import sys
import glob
from pathlib import Path

# Registro del instante exacto de inicio del script
T_SCRIPT_GLOBAL_START = time.time()

# ------------------------------------------------------------------------------
# 0. Selector de Modelos Tier 4 (Optimizados para 80 GB VRAM / HBM2e 2 TB/s)
# ------------------------------------------------------------------------------
MODELO_ELEGIDO = os.environ.get("COLAB_MODEL_TIER4", "1")

# Interruptor de Persistencia en Google Drive (Caché Permanente de Modelos GGUF)
# Si es True: monta Drive y almacena la caché de Hugging Face en tu Google Drive personal
# Si es False: descarga en disco local efímero de Colab (/root/.cache/huggingface)
USE_GOOGLE_DRIVE_CACHE = os.environ.get("USE_GOOGLE_DRIVE_CACHE", "1").lower() in ("1", "true", "yes")

if USE_GOOGLE_DRIVE_CACHE:
    try:
        from google.colab import drive
        drive_mount_path = Path("/content/drive")
        if not (drive_mount_path / "MyDrive").exists():
            print("\n💾 Montando Google Drive para reutilizar modelos en caché...")
            drive.mount(str(drive_mount_path))
        
        hf_cache_dir = drive_mount_path / "MyDrive" / "ColabModelsCache" / "huggingface"
        hf_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_cache_dir)
        print(f"✅ Google Drive conectado: Caché HF persistente en {hf_cache_dir}")
        print("   (Si el modelo ya fue descargado previamente, iniciará de inmediato sin descargar)")
    except Exception as e:
        print(f"⚠️ No se pudo montar Google Drive ({e}). Usando almacenamiento efímero local.")
        USE_GOOGLE_DRIVE_CACHE = False

CATALOGO_MODELOS = {
    "1": {
        "repo": "mudler/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-APEX-GGUF",
        "file": "Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-APEX-Quality.gguf",
        "alias": "qwen3.6-opus:apex",
        "desc": "⚡ Huihui Qwen 3.6 35B-A3B Claude 4.7 Opus APEX (1 Nov 2026 / MoE Ultra-Rápido ~130 tok/s - 262k ctx)",
        "context": "262144"
    },
    "2": {
        "repo": "mradermacher/Kimi-Dev-72B-abliterated-GGUF",
        "file": "Kimi-Dev-72B-abliterated.Q4_K_M.gguf",
        "alias": "kimi-dev-72b:uncensored",
        "desc": "💻 Kimi Dev 72B Abliterated (28 Oct 2026 / SOTA SWE & Coding Frontier / 128k ctx - 22 tok/s)",
        "context": "131072"
    },
    "3": {
        "repo": "mradermacher/GoldDiamondGold-Abliterated-L33-70b-GGUF",
        "file": "GoldDiamondGold-Abliterated-L33-70b.Q4_K_M.gguf",
        "alias": "gold-diamond-70b:uncensored",
        "desc": "💎 GoldDiamondGold 70B Abliterated (3 Nov 2026 / Post-train Frontier SWE / 128k ctx)",
        "context": "131072"
    },
    "4": {
        "repo": "unsloth/Hermes-4-70B-GGUF",
        "file": "Hermes-4-70B-Q4_K_M.gguf",
        "alias": "hermes-4-70b:uncensored",
        "desc": "🤖 Hermes 4 70B Unsloth (Nous Research / Agentes Autónomos & JSON Mode / 128k ctx)",
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
print("🚀 GOOGLE COLAB LLAMA-SERVER | TIER 4: NVIDIA A100 (80 GB HBM2e)")
print(f"🎯 Modelo activo:      {DESCRIPCION}")
print(f"📦 Repositorio GGUF:   {REPO_MODELO}")
if ARCHIVO_MODELO:
    print(f"📄 Archivo específico: {ARCHIVO_MODELO}")
print(f"📏 Ventana contexto:   {int(CONTEXT_TOKENS):,} tokens (KV Cache Q8_0 + Flash Attention)")
print("=" * 75)

# ------------------------------------------------------------------------------
# 1. Verificación de GPU NVIDIA
# ------------------------------------------------------------------------------
print("🔍 1/4 Verificando GPU NVIDIA y memoria VRAM disponible...")
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
    "-np", "2",
    "--kv-unified",
    "--cont-batching",
    "--jinja"
]

if ARCHIVO_MODELO:
    server_cmd.extend(["-hff", ARCHIVO_MODELO])

t_deps_done = time.time()
t_inicio = t_deps_done
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
t_tunnel_ready = None

for line in tunnel.stdout:
    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
    if match:
        url_encontrada = match.group(0)
        t_tunnel_ready = time.time()
        break

if not url_encontrada:
    print("❌ No se pudo obtener la URL de Cloudflare.")
else:
    t_end = t_tunnel_ready if t_tunnel_ready else time.time()
    t_deps = t_deps_done - T_SCRIPT_GLOBAL_START
    t_descarga = (t_descarga_fin - t_deps_done) if t_descarga_fin else 0.0
    t_vram = (t_fin - t_descarga_fin) if t_descarga_fin else (t_fin - t_deps_done)
    t_tunnel = (t_end - t_fin)
    t_total_global = t_end - T_SCRIPT_GLOBAL_START

    mins, secs = divmod(t_total_global, 60)
    fmt_total = f"{int(mins)}m {secs:.1f}s" if mins > 0 else f"{secs:.1f}s"

    print("\n" + "=" * 75)
    print("✅ ¡SERVIDOR LISTO CON GPU NVIDIA A100 (80 GB)!")
    print(f"🧠 Modelo:        {ALIAS_MODELO}")
    print(f"📏 Contexto:      {int(CONTEXT_TOKENS):,} tokens (Flash Attention + Q8 KV Cache)")
    print(f"🔗 URL del Túnel: {url_encontrada}")
    print("─" * 75)
    print("⏱️  TELEMETRÍA DE ARRANQUE COMPLETO (Script Start ➔ Conexión Lista):")
    print(f"   • ⚙️  Preparación y dependencias:        {t_deps:.1f} s")
    if t_descarga < 3.0:
        print(f"   • 📦 Verificación en Caché Local:      {t_descarga:.2f} s (Hit en caché)")
    else:
        print(f"   • 📥 Descarga Hugging Face:            {t_descarga:.1f} s")
    print(f"   • 🚀 Alocación en VRAM y KV Cache:     {t_vram:.1f} s")
    if t_carga_tensores:
        print(f"   • ⚡ Lectura de Tensores (llama.cpp):  {t_carga_tensores:.2f} s")
    print(f"   • 🌐 Negociación de Túnel Cloudflare:  {t_tunnel:.1f} s")
    print(f"   • 🏁 TIEMPO TOTAL HASTA DISPONIBILIDAD: {t_total_global:.1f} s ({fmt_total})")
    print("=" * 75)
    print("\n👉 EN LA TERMINAL DE TU WORKSTATION / LAPTOP, EJECUTA:\n")
    print(f"   colab-connect {url_encontrada}")
    print("\n" + "=" * 75)

import threading

def _drain_pipe(pipe, log_path):
    try:
        with open(log_path, "a", encoding="utf-8", errors="ignore") as f:
            for line in pipe:
                f.write(line)
    except Exception:
        pass

threading.Thread(target=_drain_pipe, args=(server_proc.stdout, "/tmp/llama-server.log"), daemon=True).start()
threading.Thread(target=_drain_pipe, args=(tunnel.stdout, "/tmp/cloudflared.log"), daemon=True).start()

try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("\n🛑 Servidor detenido.")
