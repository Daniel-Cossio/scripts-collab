# 🚀 Scripts de Despliegue de LLMs en Google Colab con GPU CUDA Real

Colección de scripts optimizados para ejecutar modelos de lenguaje de vanguardia (**2026**) en **Google Colab** utilizando aceleración GPU nativa mediante `llama-server` (`llama.cpp` + CUDA 12.8) y túneles cifrados de Cloudflare.

Permite conectar entornos de desarrollo locales y agentes de programación (OpenCode, Claude Code, Aider, Cursor) a la GPU remota de Colab con latencia mínima, soporte de herramientas (*tool-calling* con Jinja) y sin límites artificiales de contexto.

---

## 🛡️ Estándar de Privacidad (Zero-Leak Policy)

Este repositorio opera bajo una estricta directriz de seguridad:
* **Zero Credenciales:** Ningún archivo versionado almacena contraseñas, tokens de API, credenciales de túnel o IPs públicas reales.
* **Uso de Placeholders:** Toda referencia a instancias dinámicas o endpoints se documenta con marcadores estandarizados (`<URL_CLOUDFLARE>`, `<PUERTO_LOCAL>`, `<ALIAS_MODELO>`).
* **Archivos Locales:** Cualquier apunte personal, override o secreto de producción reside exclusivamente en `secrets-local.md` (ignorado por `.gitignore`).

---

## 📊 Matriz de Hardware y Modelos (Tiers)

| Tier | Hardware Colab | VRAM | Modelo Principal (0% Censura) | Contexto | Enfoque de Uso |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | Tesla T4 *(Gratis)* | 15.3 GB | `DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED` | **128k tokens** | Costo cero, refactorización masiva, 45 tok/s |
| **Tier 2** | NVIDIA L4 *(Pro)* | 24.0 GB | `Huihui-Qwen3.6-35B-A3B MoE` (`Q3_K_M`) | **128k tokens** | MoE ultra-rápido (~90 tok/s), SWE profesional |
| **Tier 3** | NVIDIA A100 *(40GB)* | 40.5 GB | `Llama-3.3-70B-Instruct-abliterated` (`Q3_K_M`) | **64k tokens** | Modelo frontera 70B estable sin riesgo de OOM (~45 tok/s) |
| **Tier 4** | NVIDIA A100 *(80GB)* | 81.9 GB | `Llama-3.3-70B-Instruct-abliterated` (`Q4_K_M`) | **128k tokens** | Máxima fidelidad SWE, 128k contexto, bus HBM2e 2 TB/s |

---

## 📂 Estructura de Scripts

* **`colab_server_adaptive.py` (Recomendado):**
  Script inteligente auto-detectable. Analiza la GPU asignada mediante `nvidia-smi`, selecciona automáticamente el Tier (1, 2, 3 o 4), configura los buffers de VRAM, activa Flash Attention y levanta el modelo con su contexto óptimo.
* **`colab_tier1_t4.py`:**
  Especializado para la GPU gratuita Tesla T4 (15 GB VRAM).
* **`colab_tier2_l4.py`:**
  Especializado para la GPU NVIDIA L4 (24 GB VRAM en Colab Pro).
* **`colab_tier3_a100.py`:**
  Especializado para aceleradores NVIDIA A100 40 GB (SXM4).
* **`colab_tier4_a100_80gb.py`:**
  Especializado para aceleradores NVIDIA A100 80 GB (HBM2e / High-RAM). Cuantización `Q4_K_M` con **128k tokens** de contexto y `q8_0` KV cache.

---

## 🛠️ Instrucciones de Uso

### 1. En Google Colab
1. Abre un cuaderno nuevo en [Google Colab](https://colab.research.google.com/).
2. Asegúrate de tener una GPU activa:
   * Menú: `Entorno de ejecución` -> `Cambiar tipo de entorno de ejecución`.
   * Selecciona: **T4 GPU** (o **L4 / A100** si tienes Colab Pro).
3. En una celda de código, pega y ejecuta el script deseado (por ejemplo, `colab_server_adaptive.py`):
   ```python
   # Opcional: clonar el repositorio directamente en la celda
   !git clone https://github.com/Daniel-Cossio/scripts-collab.git
   %run -i -e -d /content/scripts-collab/colab_server_adaptive.py
   ```
   *(O simplemente copia y pega el contenido completo del archivo en la celda).*

### 2. En tu Estación de Trabajo Local (Cliente)
Al finalizar el arranque en Colab, el script imprimirá en consola una URL segura generada por Cloudflare:
```text
https://<subdominio-aleatorio>.trycloudflare.com
```

Copia esa URL y ejecútala en tu terminal local con el helper configurado:
```bash
colab-connect https://<subdominio-aleatorio>.trycloudflare.com
```

Esto conectará el proxy local `colab-bridge` (puerto `8898`), permitiendo que cualquier herramienta local apunte a `http://127.0.0.1:8898/v1` como si fuera un servidor OpenAI estándar.

---

## ⚙️ Características Técnicas del Runtime

* **Motor:** `llama-server` oficial precompilado con soporte CUDA 12.8 y cuBLAS.
* **Flash Attention:** Activado explícitamente (`--flash-attn on`) para reducir el footprint cuadrático de la atención.
* **KV Cache Quantization:** Claves y valores cuantizados en 8 bits (`-ctk q8_0 -ctv q8_0`), lo que permite duplicar el contexto con degradación cero de precisión.
* **Tool Calling / Jinja Parser:** Parámetro `--jinja` activo para permitir llamadas de funciones estructuradas requeridas por agentes de programación autónomos.
* **Sin proyectores innecesarios:** Flag `--no-mmproj` para evitar reservar VRAM en pesos multimodales cuando se opera con código y texto puro.

---

## 📜 Licencia y Mantenimiento

Desarrollado y mantenido por Daniel Cossio para automatización de flujos de IA privada y soberana.
Sujeto a las directrices de la Constitución de Agentes.
