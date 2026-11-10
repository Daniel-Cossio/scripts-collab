#!/usr/bin/env python3
"""
Ejecutor de Pre-warm desatendido en Google Colab (CPU - Tier 0)
Descarga modelos pesados a Google Drive (5 TB) con costo de cómputo mínimo (~0.08 CU/hr)
y garantiza el apagado inmediato (Zero-Waste Policy).
"""

import sys
import time
import subprocess
from pathlib import Path

SESSION_NAME = f"prewarm-heavy-{int(time.time())}"
SCRIPTS_DIR = Path("/home/ubuntu/Documentos/git/scripts-collab")
RCLONE_CONF = Path.home() / ".config/rclone/rclone.conf"

def get_colab_balance():
    try:
        out = subprocess.check_output(["colab", "usage"], text=True)
        balance, rate = None, None
        for line in out.splitlines():
            if "Current balance:" in line:
                balance = float(line.split(":")[1].replace("compute units", "").strip())
            elif "Usage rate:" in line:
                rate = float(line.split(":")[1].replace("/hr", "").strip())
        return balance, rate
    except Exception:
        return None, None

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "heavy"
    print("=" * 75)
    print(f"🚀 INICIANDO PRE-WARM EN COLAB CPU (Tier 0): {target.upper()}")
    print(f"   Sesión: {SESSION_NAME}")
    print("=" * 75)

    initial_balance, _ = get_colab_balance()
    t_start = time.time()

    try:
        # 1. Crear sesión CPU
        print(f"⏳ [1/5] Creando sesión CPU '{SESSION_NAME}' (0.08 CU/hr)...")
        subprocess.run(["colab", "new", "-s", SESSION_NAME], check=True)
        print("   ✅ Sesión CPU lista.")

        # 2. Subir scripts y configuración
        print("📤 [2/5] Subiendo rclone.conf, setup_drive_headless.py y prewarm_models.py...")
        subprocess.run(["colab", "upload", "-s", SESSION_NAME, str(RCLONE_CONF), "/content/rclone.conf"], check=True)
        subprocess.run(["colab", "upload", "-s", SESSION_NAME, str(SCRIPTS_DIR / "setup_drive_headless.py"), "/content/setup_drive_headless.py"], check=True)
        subprocess.run(["colab", "upload", "-s", SESSION_NAME, str(SCRIPTS_DIR / "prewarm_models.py"), "/content/prewarm_models.py"], check=True)
        print("   ✅ Archivos transferidos a la VM.")

        # 3. Iniciar descarga en background dentro de la VM
        print(f"📥 [3/5] Lanzando prewarm_models.py {target} en background...")
        launch_code = f"""
import subprocess
import os

proc = subprocess.Popen(
    ["python3", "-u", "/content/prewarm_models.py", "{target}"],
    stdout=open("/content/prewarm.log", "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True
)
print(f"PID:{{proc.pid}}")
"""
        res = subprocess.check_output(
            ["colab", "exec", "-s", SESSION_NAME, "--timeout", "30"],
            input=launch_code,
            text=True
        ).strip()
        print(f"   ✅ Proceso iniciado: {res}")

        # 4. Monitorear progreso
        print("👀 [4/5] Monitoreando progreso de descarga (espera activa cada 20s)...")
        monitor_code = """
import os, subprocess

is_running = False
try:
    # Chequear si python3 prewarm_models.py sigue en ejecucion
    ps = subprocess.check_output("ps aux | grep prewarm_models.py | grep -v grep", shell=True, text=True)
    if "python3" in ps:
        is_running = True
except Exception:
    is_running = False

log_tail = ""
if os.path.exists("/content/prewarm.log"):
    try:
        log_tail = subprocess.check_output(["tail", "-n", "8", "/content/prewarm.log"], text=True)
    except Exception:
        pass

print(f"STATUS:{is_running}---LOG---{log_tail}")
"""
        max_duration = 7200  # 120 minutos max
        start_monitor = time.time()
        last_log = ""

        while time.time() - start_monitor < max_duration:
            time.sleep(20)
            try:
                mon_out = subprocess.check_output(
                    ["colab", "exec", "-s", SESSION_NAME, "--timeout", "30"],
                    input=monitor_code,
                    text=True
                ).strip()

                if "---LOG---" in mon_out:
                    parts = mon_out.split("---LOG---")
                    running_str = parts[0].replace("STATUS:", "").strip()
                    current_log = parts[1].strip()

                    if current_log != last_log:
                        print(f"\n--- [Progreso @ {int(time.time() - t_start)}s] ---")
                        print(current_log)
                        last_log = current_log

                    if "Proceso de pre-warm" in current_log and "finalizado" in current_log:
                        print("\n🎉 Pre-warm y sincronización a Google Drive completados exitosamente en la VM.")
                        break

                    if running_str == "False" and ("Proceso de pre-warm" not in current_log or "finalizado" not in current_log):
                        print("\n⚠️ El proceso en la VM terminó antes de lo esperado.")
                        break
            except Exception as e:
                print(f"⚠️ Aviso en monitoreo: {e}")

        # 5. Verificación final de archivos en Drive directamente vía Rclone
        print("\n🔍 [5/5] Verificando presencia de modelos en Google Drive vía Rclone local...")
        try:
            drive_check = subprocess.check_output(
                ["rclone", "ls", "gdrive:ColabModelsCache/huggingface/hub/"],
                text=True
            ).strip()
            print("   📁 Archivos confirmados en Caché Drive:")
            for line in drive_check.splitlines():
                if any(ext in line for ext in [".gguf", "refs", "blobs"]):
                    print(f"      {line}")
        except Exception as e:
            print(f"   ⚠️ No se pudo verificar directamente en Drive: {e}")

    except Exception as e:
        print(f"\n❌ Error durante el proceso de pre-warm: {e}")

    finally:
        # Zero-Waste: Apagar la sesión
        print(f"\n🛑 Apagando y liberando sesión CPU '{SESSION_NAME}' (Zero-Waste Policy)...")
        try:
            subprocess.run(["colab", "stop", "-s", SESSION_NAME], check=True)
            print("   ✅ Sesión detenida con éxito.")
        except Exception as e:
            print(f"   ⚠️ Error al detener sesión: {e}")

        final_balance, final_rate = get_colab_balance()
        t_total = time.time() - t_start
        print(f"⏱️ Tiempo total de ejecución: {t_total:.1f}s ({t_total/60:.2f} min)")
        if initial_balance is not None and final_balance is not None:
            spent = round(initial_balance - final_balance, 4)
            print(f"💰 CUs gastadas: {spent} CU (Consumo actual en reposo: {final_rate}/hr)")

if __name__ == "__main__":
    main()
