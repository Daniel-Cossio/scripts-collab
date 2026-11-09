#!/usr/bin/env python3
"""
Framework de Orquestación y Evaluación Comparativa Multi-Tier en Google Colab
Mide empíricamente:
- Tiempos de provisión, carga a VRAM desde Drive (5 TB)
- Latencia de respuesta, TTFT, Throughput (tokens/segundo)
- Consumo de ventanas de contexto (Prompt / Completion tokens)
- Consumo real de Compute Units (CUs)
- Calidad técnica objetiva: Análisis AST, ejecución de pytest sobre código generado
- Cierre forzoso de instancias bajo política Zero-Waste
"""

import os
import sys
import time
import json
import ast
import subprocess
import argparse
from pathlib import Path

BASE_DIR = Path("/home/ubuntu/Documentos/git/scripts-collab")
OUTPUT_DIR = BASE_DIR / "eval_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RCLONE_CONF = Path.home() / ".config/rclone/rclone.conf"
RESULTS_FILE = BASE_DIR / "benchmark_comparison_results.json"

TASK_PROMPT = """Implement an asynchronous Token Bucket rate limiter in Python.

Requirements:
1. Define class `AsyncTokenBucket`:
   - `__init__(self, capacity: float, refill_rate: float)`: capacity is max tokens, refill_rate is tokens added per second.
   - `async def acquire(self, tokens: float = 1.0) -> bool`: Consumes tokens immediately if available and returns True. If not enough tokens, returns False without waiting.
   - `async def wait_for_tokens(self, tokens: float = 1.0, timeout: float = None) -> bool`: Waits asynchronously until enough tokens are available to consume, then consumes them and returns True. If timeout expires before tokens are available, returns False.
   - Ensure coroutine-safety using `asyncio.Lock`.
2. Write a unit test suite using `pytest` and `pytest-asyncio` testing:
   - Immediate acquire succeeds when tokens available.
   - Immediate acquire fails when tokens exhausted.
   - Tokens refill correctly over time.
   - `wait_for_tokens` successfully waits and acquires when refilled.
   - `wait_for_tokens` returns False on timeout.
   - Concurrent tasks acquiring tokens safely.

Provide clean, modular, typed Python code with all tests included. Output executable python code inside ```python blocks."""

TIERS_CONFIG = {
    "1": {
        "name": "tier1-t4",
        "accelerator": "T4",
        "high_mem": False,
        "script": "colab_tier1_t4.py",
        "model_alias": "deepseek-v4-qwen3.5:9b",
        "context_tokens": "131072",
        "cost_hr": 1.15
    },
    "2": {
        "name": "tier2-l4",
        "accelerator": "L4",
        "high_mem": False,
        "script": "colab_tier2_l4.py",
        "model_alias": "qwen3.6-35b-a3b:uncensored",
        "context_tokens": "131072",
        "cost_hr": 1.94
    },
    "3": {
        "name": "tier3-a100-40gb",
        "accelerator": "A100",
        "high_mem": False,
        "script": "colab_tier3_a100.py",
        "model_alias": "kimi-linear-48b-a3b:uncensored",
        "context_tokens": "131072",  # Ajustado para carga balanceada en 40GB
        "cost_hr": 6.73
    },
    "4": {
        "name": "tier4-a100-80gb",
        "accelerator": "A100",
        "high_mem": True,
        "script": "colab_tier4_a100_80gb.py",
        "model_alias": "qwen3.6-opus:apex",
        "context_tokens": "262144",
        "cost_hr": 8.88
    }
}

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

def evaluate_code_quality(output_text: str, tier_id: str) -> dict:
    """Extrae código python, valida AST y ejecuta pytest sobre las pruebas generadas."""
    eval_res = {
        "ast_valid": False,
        "lines_of_code": 0,
        "tests_collected": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "has_lock": False,
        "has_acquire": False,
        "has_wait": False,
        "all_tests_pass": False,
        "error": None
    }
    
    # Extraer bloques de python
    code_blocks = []
    in_block = False
    current_block = []
    for line in output_text.splitlines():
        if line.strip().startswith("```python"):
            in_block = True
            current_block = []
            continue
        elif in_block and line.strip().startswith("```"):
            in_block = False
            code_blocks.append("\n".join(current_block))
            current_block = []
            continue
        if in_block:
            current_block.append(line)
            
    if not code_blocks:
        code_to_eval = output_text
    else:
        code_to_eval = "\n\n".join(code_blocks)
        
    eval_res["lines_of_code"] = len(code_to_eval.splitlines())
    eval_res["has_lock"] = "asyncio.Lock" in code_to_eval or "Lock(" in code_to_eval
    eval_res["has_acquire"] = "def acquire" in code_to_eval
    eval_res["has_wait"] = "def wait_for_tokens" in code_to_eval

    # Validar AST
    try:
        ast.parse(code_to_eval)
        eval_res["ast_valid"] = True
    except SyntaxError as se:
        eval_res["ast_valid"] = False
        eval_res["error"] = f"SyntaxError: {se}"
        return eval_res

    # Guardar archivo para pytest
    test_file = OUTPUT_DIR / f"test_tier_{tier_id}.py"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(code_to_eval)

    # Ejecutar pytest con uv
    try:
        cmd = ["uv", "tool", "run", "--with", "pytest-asyncio", "pytest", str(test_file), "-q", "--tb=short"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        pytest_out = proc.stdout + proc.stderr
        
        # Parsear salida de pytest
        for line in pytest_out.splitlines():
            if "passed" in line or "failed" in line or "error" in line:
                import re
                p_match = re.search(r"(\d+)\s+passed", line)
                f_match = re.search(r"(\d+)\s+failed", line)
                if p_match:
                    eval_res["tests_passed"] = int(p_match.group(1))
                if f_match:
                    eval_res["tests_failed"] = int(f_match.group(1))
        
        eval_res["tests_collected"] = eval_res["tests_passed"] + eval_res["tests_failed"]
        if eval_res["tests_collected"] > 0 and eval_res["tests_failed"] == 0:
            eval_res["all_tests_pass"] = True
        eval_res["pytest_output_summary"] = pytest_out.strip().splitlines()[-1] if pytest_out.strip() else ""
    except Exception as e:
        eval_res["error"] = f"Pytest execution error: {e}"

    return eval_res

def run_single_tier_benchmark(tier_id: str) -> dict:
    tier_info = TIERS_CONFIG[str(tier_id)]
    session_name = f"eval-t{tier_id}-{int(time.time())}"
    accelerator = tier_info["accelerator"]
    high_mem = tier_info["high_mem"]
    script_name = tier_info["script"]

    print("\n" + "=" * 80)
    print(f"🔬 INICIANDO EVALUACIÓN MULTI-TIER: Tier {tier_id} ({tier_info['name']})")
    print(f"   Acelerador: {accelerator} | High-Mem: {high_mem} | Modelo: {tier_info['model_alias']}")
    print("=" * 80)

    initial_balance, _ = get_colab_balance()
    t_start = time.time()
    
    result = {
        "tier_id": tier_id,
        "name": tier_info["name"],
        "accelerator": accelerator,
        "high_mem": high_mem,
        "model": tier_info["model_alias"],
        "cost_hr": tier_info["cost_hr"],
        "status": "INITIALIZING",
        "metrics": {},
        "quality": {},
        "timestamps": {"start": t_start}
    }

    try:
        # 1. Provisión de la VM
        t0 = time.time()
        new_cmd = ["colab", "new", "-s", session_name, "--gpu", accelerator]
        if high_mem:
            new_cmd.append("--high-mem")
        print(f"⏳ [1/6] Provisionando VM con {' '.join(new_cmd)}...")
        try:
            colab_env = os.environ.copy()
            colab_env["COLAB_CLI_ASSIGN_RETRIES"] = "1"
            colab_env["COLAB_CLI_RETRY_BACKOFF"] = "3"
            subprocess.run(new_cmd, check=True, env=colab_env, timeout=120)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"⚠️ Acelerador {accelerator} no disponible actualmente en el pool: {e}")
            result["status"] = "UNAVAILABLE"
            result["error"] = f"GPU pool exhausted for {accelerator}: {e}"
            return result
            
        t_provision = time.time() - t0
        result["metrics"]["t_provision_sec"] = round(t_provision, 2)
        print(f"   ✅ VM Provisionada en {t_provision:.2f}s")

        # 2. Consultar GPU Real
        print("🔍 [2/6] Verificando GPU en la VM...")
        gpu_info = subprocess.check_output(
            ["colab", "exec", "-s", session_name, "--timeout", "30"],
            input="import subprocess; print(subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip())",
            text=True
        ).strip()
        result["metrics"]["gpu_hardware"] = gpu_info
        print(f"   🎮 GPU Real: {gpu_info}")

        # 3. Subir scripts y montar Google Drive
        print("📤 [3/6] Transfiriendo rclone.conf y scripts a la VM...")
        subprocess.run(["colab", "upload", "-s", session_name, str(RCLONE_CONF), "/content/rclone.conf"], check=True)
        subprocess.run(["colab", "upload", "-s", session_name, str(BASE_DIR / "setup_drive_headless.py"), "/content/setup_drive_headless.py"], check=True)
        subprocess.run(["colab", "upload", "-s", session_name, str(BASE_DIR / script_name), f"/content/{script_name}"], check=True)

        # 4. Lanzar llama-server en background dentro de la VM
        print(f"🚀 [4/6] Iniciando {script_name} y cargando modelo desde Drive...")
        t_load_start = time.time()
        launch_code = f"""
import subprocess, os
env = os.environ.copy()
env["COLAB_CONTEXT_TOKENS"] = "{tier_info['context_tokens']}"
proc = subprocess.Popen(
    ["python3", "-u", "/content/{script_name}"],
    stdout=open("/content/server.log", "w"),
    stderr=subprocess.STDOUT,
    env=env,
    start_new_session=True
)
print(f"PID:{{proc.pid}}")
"""
        subprocess.check_output(["colab", "exec", "-s", session_name, "--timeout", "30"], input=launch_code, text=True)

        # Esperar a que el servidor responda en http://localhost:8899/v1/models
        server_ready = False
        max_wait_server = 180  # 3 minutos max
        start_wait = time.time()
        check_code = """
import urllib.request, json
try:
    with urllib.request.urlopen("http://localhost:8899/v1/models", timeout=2) as resp:
        print("READY")
except Exception as e:
    print(f"WAITING: {e}")
"""
        while time.time() - start_wait < max_wait_server:
            time.sleep(4)
            probe = subprocess.check_output(["colab", "exec", "-s", session_name, "--timeout", "10"], input=check_code, text=True).strip()
            if "READY" in probe:
                server_ready = True
                break

        if not server_ready:
            print("❌ Error: llama-server no alcanzó el estado READY dentro del tiempo límite.")
            # Obtener tail del log
            log_tail = subprocess.check_output(["colab", "exec", "-s", session_name, "--timeout", "10"], input="import subprocess; print(subprocess.check_output(['tail', '-n', '25', '/content/server.log'], text=True))", text=True)
            print(f"   Log del servidor:\n{log_tail}")
            result["status"] = "FAILED"
            result["error"] = f"Server startup timeout. Log:\n{log_tail}"
            return result

        t_load_done = time.time() - t_load_start
        result["metrics"]["t_load_sec"] = round(t_load_done, 2)
        print(f"   ✅ llama-server listo en {t_load_done:.2f}s (Hit en Drive + VRAM)")

        # Medir VRAM tras la carga
        vram_used = subprocess.check_output(
            ["colab", "exec", "-s", session_name, "--timeout", "10"],
            input="import subprocess; print(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader'], text=True).strip())",
            text=True
        ).strip()
        result["metrics"]["vram_used_after_load"] = vram_used
        print(f"   🧠 VRAM Ocupada: {vram_used}")

        # 5. Delegar y Ejecutar Tarea de Benchmark Idéntica
        print("🎯 [5/6] Delegando tarea de benchmark idéntica al modelo...")
        payload_data = {
            "model": tier_info["model_alias"],
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Python engineer. Implement the requested Python module and pytest test suite. Output executable python code inside ```python blocks without fluff."
                },
                {"role": "user", "content": TASK_PROMPT}
            ],
            "temperature": 0.2,
            "max_tokens": 2048,
            "stream": False
        }

        exec_code = f"""
import urllib.request, json, time

payload = {json.dumps(payload_data)}
data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request("http://localhost:8899/v1/chat/completions", data=data, headers={{"Content-Type": "application/json"}})

t0 = time.time()
with urllib.request.urlopen(req, timeout=300) as resp:
    res = json.loads(resp.read().decode("utf-8"))
t_total = time.time() - t0

res["__client_timing"] = {{"total_sec": t_total}}
with open("/content/response.json", "w", encoding="utf-8") as f:
    json.dump(res, f)

print(json.dumps({{
    "total_sec": t_total,
    "usage": res.get("usage", {{}}),
    "finish_reason": res["choices"][0].get("finish_reason", "")
}}))
"""
        response_summary_str = subprocess.check_output(
            ["colab", "exec", "-s", session_name, "--timeout", "360"],
            input=exec_code,
            text=True
        ).strip()

        resp_meta = json.loads(response_summary_str.splitlines()[-1])
        t_generation = resp_meta.get("total_sec", 0.0)
        usage = resp_meta.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        tok_per_sec = round(completion_tokens / t_generation, 2) if t_generation > 0 else 0.0

        result["metrics"]["t_generation_sec"] = round(t_generation, 2)
        result["metrics"]["prompt_tokens"] = prompt_tokens
        result["metrics"]["completion_tokens"] = completion_tokens
        result["metrics"]["total_tokens"] = total_tokens
        result["metrics"]["tokens_per_second"] = tok_per_sec
        result["metrics"]["finish_reason"] = resp_meta.get("finish_reason")

        print(f"   ⚡ Generación: {completion_tokens} tokens en {t_generation:.2f}s ➔ {tok_per_sec} tok/s")

        # Descargar el contenido generado a local
        read_resp_code = "import json; res = json.load(open('/content/response.json')); print(res['choices'][0]['message']['content'])"
        generated_code_text = subprocess.check_output(
            ["colab", "exec", "-s", session_name, "--timeout", "30"],
            input=read_resp_code,
            text=True
        )

        # 6. Evaluación de Calidad y Tests en Local
        print("🧪 [6/6] Ejecutando análisis de AST y suite de pytest sobre el código...")
        quality = evaluate_code_quality(generated_code_text, tier_id)
        result["quality"] = quality
        print(f"   📝 Líneas de código: {quality['lines_of_code']} | AST Válido: {quality['ast_valid']}")
        print(f"   🚦 Tests Ejecutados: {quality['tests_collected']} | Pasados: {quality['tests_passed']} | Fallados: {quality['tests_failed']}")
        print(f"   🏆 ¿Logró el Objetivo?: {'SÍ' if quality['all_tests_pass'] else 'PARCIAL / NO'}")

        result["status"] = "SUCCESS"

    except Exception as e:
        print(f"❌ Error durante la evaluación de Tier {tier_id}: {e}")
        result["status"] = "FAILED"
        result["error"] = str(e)

    finally:
        # Zero-Waste: Apagar la sesión inmediatamente
        print(f"🛑 [Zero-Waste] Apagando y liberando sesión {session_name}...")
        t_teardown_start = time.time()
        try:
            subprocess.run(["colab", "stop", "-s", session_name], check=True)
            print("   ✅ Sesión Colab apagada.")
        except Exception:
            pass
        t_teardown = time.time() - t_teardown_start
        result["metrics"]["t_teardown_sec"] = round(t_teardown, 2)

        final_balance, final_rate = get_colab_balance()
        if initial_balance is not None and final_balance is not None:
            spent = round(initial_balance - final_balance, 4)
            result["metrics"]["cu_spent"] = spent
            print(f"   💰 CUs gastadas en la prueba: {spent} CU (Consumo en reposo: {final_rate}/hr)")

    return result

def main():
    parser = argparse.ArgumentParser(description="Multi-Tier Orchestration & Benchmark Evaluator")
    parser.add_argument("--tier", "-t", default="all", choices=["1", "2", "3", "4", "all"], help="Tier específico o 'all'")
    args = parser.parse_args()

    tiers_to_run = ["1", "2", "3", "4"] if args.tier == "all" else [args.tier]
    
    print("\n" + "#" * 80)
    print("🏆 BATERÍA DE EVALUACIÓN MULTI-TIER: ARBITRAJE DE CÓMPUTO Y CALIDAD")
    print(f"   Tiers en cola: {', '.join(tiers_to_run)}")
    print("#" * 80)

    all_results = []
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        except Exception:
            all_results = []

    for t in tiers_to_run:
        res = run_single_tier_benchmark(t)
        all_results.append(res)
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        # Pausa de 3 segundos entre tiers para asegurar desconexión de red
        time.sleep(3)

    print("\n" + "=" * 80)
    print("🏁 EVALUACIÓN MULTI-TIER FINALIZADA CON ÉXITO")
    print(f"   Resultados consolidados en: {RESULTS_FILE}")
    print("=" * 80)

if __name__ == "__main__":
    main()
