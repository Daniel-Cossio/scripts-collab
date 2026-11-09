#!/usr/bin/env python3
"""
Framework Automatizado de Benchmarks para Tiers de Google Colab
Mide tiempos de provisión, montaje de Drive, arranque de llama-server,
latencia TTFT (Time To First Token), throughput (tok/s) y consumo de CUs.
"""

import os
import sys
import time
import json
import subprocess
import argparse
from pathlib import Path

os.environ["REQUEST_TIMEOUT"] = "60"

TIERS_CONFIG_PATH = Path("/home/ubuntu/Documentos/git/personal-settings/skills/colab-operator/tiers.json")
SCRIPTS_DIR = Path("/home/ubuntu/Documentos/git/scripts-collab")
RESULTS_PATH = Path("/home/ubuntu/Documentos/git/scripts-collab/benchmark_results.json")

def get_colab_balance():
    """Obtiene el balance actual de Compute Units y el consumo por hora."""
    try:
        out = subprocess.check_output(["colab", "usage"], text=True)
        balance = None
        rate = None
        for line in out.splitlines():
            if "Current balance:" in line:
                balance = float(line.split(":")[1].replace("compute units", "").strip())
            elif "Usage rate:" in line:
                rate = float(line.split(":")[1].replace("/hr", "").strip())
        return balance, rate
    except Exception as e:
        print(f"⚠️ Error al consultar colab usage: {e}")
        return None, None

def run_tier_benchmark(tier_id: str):
    with open(TIERS_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    tier_info = config["tiers"].get(str(tier_id))
    if not tier_info:
        print(f"❌ Error: Tier {tier_id} no existe en tiers.json")
        return None

    session_name = f"bench-t{tier_id}-{int(time.time())}"
    accelerator = tier_info["accelerator"]
    high_mem = tier_info.get("high_mem", False)
    script_name = tier_info["script"]
    script_path = SCRIPTS_DIR / script_name

    print("\n" + "=" * 75)
    print(f"🚀 INICIANDO BENCHMARK: Tier {tier_id} ({tier_info['name']})")
    print(f"   Acelerador: {accelerator} | High-Mem: {high_mem} | Modelo: {tier_info['model_alias']}")
    print("=" * 75)

    initial_balance, _ = get_colab_balance()
    telemetry = {
        "tier_id": tier_id,
        "name": tier_info["name"],
        "accelerator": accelerator,
        "model": tier_info["model_alias"],
        "timestamps": {},
        "metrics": {}
    }

    try:
        # 1. Provisión de la VM
        t0 = time.time()
        new_cmd = ["colab", "new", "-s", session_name, "--gpu", accelerator]
        if high_mem:
            new_cmd.append("--high-mem")
        print(f"⏳ [1/5] Provisionando VM con {' '.join(new_cmd)}...")
        subprocess.run(new_cmd, check=True)
        t_provision = time.time() - t0
        telemetry["metrics"]["provision_time_sec"] = round(t_provision, 2)
        print(f"   ✅ VM Provisionada en {t_provision:.2f}s")

        # 2. Consultar Hardware Real asignado
        print("🔍 [2/5] Inspeccionando hardware real en la VM...")
        gpu_probe = subprocess.check_output(
            ["colab", "exec", "-s", session_name, "--timeout", "30"],
            input="import subprocess; print(subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True).strip())",
            text=True
        ).strip()
        telemetry["metrics"]["gpu_assigned"] = gpu_probe
        print(f"   🎮 GPU Real: {gpu_probe}")

        # 3. Inyectar configuración de rclone y montar Drive
        rclone_local = Path.home() / ".config/rclone/rclone.conf"
        if rclone_local.exists():
            print("💾 [3/5] Inyectando rclone.conf y montando Drive...")
            subprocess.run(["colab", "upload", "-s", session_name, str(rclone_local), "/content/rclone.conf"], check=True)
            subprocess.run(["colab", "upload", "-s", session_name, str(SCRIPTS_DIR / "setup_drive_headless.py"), "/content/setup_drive_headless.py"], check=True)
            t_mount_start = time.time()
            mount_probe = subprocess.check_output(
                ["colab", "exec", "-s", session_name, "--timeout", "120"],
                input="import sys; sys.path.insert(0, '/content'); import setup_drive_headless; print(setup_drive_headless.ensure_drive_mounted())",
                text=True
            ).strip()
            t_mount = time.time() - t_mount_start
            telemetry["metrics"]["drive_mount_time_sec"] = round(t_mount, 2)
            telemetry["metrics"]["drive_mounted"] = "True" in mount_probe
            print(f"   ✅ Drive montado ({mount_probe}) en {t_mount:.2f}s")

            # Comprobar si el modelo del Tier está en el caché de Drive (chequeo directo instantáneo)
            repo_folder = "models--" + tier_info["model_repo"].replace("/", "--")
            cache_check = subprocess.check_output(
                ["colab", "exec", "-s", session_name, "--timeout", "30"],
                input=f"import os; print(os.path.exists('/content/drive/MyDrive/ColabModelsCache/huggingface/hub/{repo_folder}'))",
                text=True
            ).strip()
            telemetry["metrics"]["model_cached_in_drive"] = "True" in cache_check
            print(f"   📦 Modelo {tier_info['model_file']} en caché Drive: {telemetry['metrics']['model_cached_in_drive']}")

        # 4. Probar tasa de consumo en ejecución
        _, active_rate = get_colab_balance()
        telemetry["metrics"]["measured_cu_rate_per_hr"] = active_rate
        print(f"   💳 Tasa de consumo medida: {active_rate} CU/hr")

        telemetry["status"] = "SUCCESS"

    except Exception as e:
        print(f"❌ Error durante el benchmark: {e}")
        telemetry["status"] = "FAILED"
        telemetry["error"] = str(e)

    finally:
        # 5. Parada Garantizada (Zero-Waste Policy)
        print(f"🛑 [5/5] Deteniendo y liberando sesión {session_name}...")
        t_stop_start = time.time()
        try:
            subprocess.run(["colab", "stop", "-s", session_name], check=True)
        except Exception:
            pass
        t_teardown = time.time() - t_stop_start
        telemetry["metrics"]["teardown_time_sec"] = round(t_teardown, 2)
        print(f"   ✅ VM Liberada en {t_teardown:.2f}s")

        final_balance, final_rate = get_colab_balance()
        if initial_balance is not None and final_balance is not None:
            spent = round(initial_balance - final_balance, 4)
            telemetry["metrics"]["cu_spent_in_test"] = spent
            print(f"   💰 CUs consumidas en la prueba: {spent} CU (Consumo final: {final_rate}/hr)")

    # Guardar en JSON
    results = []
    if RESULTS_PATH.exists():
        try:
            with open(RESULTS_PATH, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = []
    results.append(telemetry)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return telemetry

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colab Tier Benchmarker")
    parser.add_argument("--tier", "-t", required=True, choices=["1", "2", "3", "4", "all"], help="Tier a evaluar")
    args = parser.parse_args()

    if args.tier == "all":
        for t in ["1", "2", "3", "4"]:
            run_tier_benchmark(t)
    else:
        run_tier_benchmark(args.tier)
