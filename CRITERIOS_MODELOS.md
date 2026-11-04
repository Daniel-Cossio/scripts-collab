# 🎯 Criterios de Selección de LLMs para Entorno Agéntico (Google Colab + OpenCode)

Este documento sirve como **guía de referencia y plantilla de búsqueda** para localizar nuevos modelos en Hugging Face, Reddit (r/LocalLLaMA), Discord, Twitter o motores de IA (Perplexity, ChatGPT, Claude).

---

## 📌 1. Los 9 Requisitos Innegociables

1. **Cero Censura (0% Refusals):**
   * Etiquetas: `abliterated`, `uncensored`, `heretic-ara`, `unaligned`.
   * Capacidad total en ciberseguridad, ingeniería inversa, sockets, kernel y pentesting sin bloqueos morales.
2. **Tool Calling / Function Calling Estricto:**
   * Compatibilidad completa con esquemas JSON estructurados vía `--jinja` en `llama-server`.
   * Cero fugas de sintaxis JSON en la respuesta textual al usuario.
3. **Capacidad de Orquestación y Subagentes:**
   * Razonamiento jerárquico para delegar tareas complejas (`invoke_subagent`).
   * Capacidad de ingestar respuestas de subagentes sin caer en bucles de re-delegación.
4. **Contexto Masivo Real (Mínimo 128k tokens):**
   * Soporte nativo para ventanas de **131,072 (128k)** a **262,144 (262k)** o **524,288 (512k)** tokens.
   * Eficiencia con Flash Attention y KV Cache cuantizado (`q8_0` o `f16`).
5. **Velocidad Interactiva (>70 a 130 tokens/segundo):**
   * **Prioridad a arquitecturas MoE** (Mezcla de Expertos, ej. 35B-A3B o 48B-A3B) donde solo se activan 2–3B parámetros por token.
   * Evitar modelos densos masivos (como 70B/72B densos a 20 tok/s) para pair-programming ágil.
6. **Vanguardia en Ingeniería de Software (2026):**
   * Modelos recientes con alta puntuación en SWE-bench, HumanEval y razonamiento de código.
   * Destilaciones de frontier (Claude 4.7 Opus, DeepSeek V4, Qwen 3.6, Kimi Dev).
7. **Formato GGUF Cuantizado (`llama.cpp` + CUDA):**
   * Archivos listos para ejecutar con `llama-server` bajo CUDA 12.8 (`Q4_K_M`, `Q6_K`, `i1-Q4_K_M`).
8. **Ajuste Quirúrgico de VRAM (Cero Errores OOM):**
   * El modelo + contexto + KV cache deben respetar estrictamente el techo de VRAM del Tier asignado.
9. **Soberanía y Compatibilidad con Túneles:**
   * Endpoint compatible con la API OpenAI `/v1/chat/completions` para operar con `colab-bridge`.

---

## 📊 2. Matriz de Restricciones por Hardware (Tiers de Colab)

| Tier | Hardware Colab | VRAM Máxima | Límite Peso GGUF | Contexto Objetivo | Perfil de Modelo Buscado |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | Tesla T4 *(Gratis)* | **15.3 GB** | **<= 6.5 GB** | **128k tokens** | 7B a 9B cuantizado (`Q4_K_M`), alta velocidad (45+ tok/s). |
| **Tier 2** | NVIDIA L4 *(Pro)* | **24.0 GB** | **<= 18.0 GB** | **128k tokens** | MoE 30B–35B (`Q3_K_M`), SWE profesional (70–90 tok/s). |
| **Tier 3** | NVIDIA A100 *(40GB)* | **40.5 GB** | **<= 30.0 GB** | **262k – 512k** | MoE con atención lineal o MoE mediano con contexto gigante. |
| **Tier 4** | NVIDIA A100 *(80GB)* | **81.9 GB** | **<= 50.0 GB** | **262k tokens** | MoE Frontier en `Q6_K` (130 tok/s) o denso 70B en `Q4_K_M`. |

---

## 🤖 3. Prompts Listos para Copiar y Pegar en otras Herramientas

### Opción A: En Español (Para ChatGPT, Perplexity o Claude)

> "Actúa como un arquitecto experto en LLMs y hardware de inferencia local. Necesito que me recomiendes los mejores modelos de lenguaje de vanguardia (**2026**) en formato **GGUF** para ejecutar en **Google Colab** con **llama-server** y conectar a mi agente de programación (**OpenCode**).
> 
> Los modelos deben cumplir estrictamente con los siguientes requisitos innegociables:
> 1. **Cero Censura / 0% Refusals:** Modelos completamente desensurados (abliterated o uncensored) para ciberseguridad, ingeniería inversa y programación de bajo nivel sin rechazos.
> 2. **Soporte de Tool Calling y Subagentes:** Compatibilidad nativa con function-calling vía templates Jinja, excelente rigor emitiendo JSON y capacidad cognitiva para lanzar y coordinar subagentes sin entrar en bucles.
> 3. **Contexto Masivo Real:** Soporte nativo para 128k tokens (mínimo) o 262k tokens.
> 4. **Velocidad y Arquitectura MoE:** Preferencia total por modelos MoE (como 35B con 3B activos) que generen a más de 70–130 tokens/segundo en GPU, evitando modelos densos lentos de 70B (20 tok/s).
> 5. **Hardware objetivo:**
>    - Para **24 GB VRAM (NVIDIA L4)**: Peso GGUF <= 18 GB.
>    - Para **80 GB VRAM (NVIDIA A100)**: Peso GGUF <= 35–45 GB en Q6_K con 262k contexto.
> 
> Dame los nombres exactos de los repositorios en Hugging Face, el autor del GGUF (ej. mradermacher, bartowski, mudler), el tipo de cuantización recomendada y un resumen de por qué superan a las alternativas actuales."

---

### Opción B: En Inglés (Para Hugging Face, Reddit r/LocalLLaMA, Discord o Perplexity)

> "I am looking for the best frontier coding & agentic LLMs released in **2026** in **GGUF format** to run on **llama-server (CUDA)** inside Google Colab and connect to **OpenCode**.
> 
> Requirements:
> 1. **100% Uncensored / Abliterated:** Must have 0% refusals for technical tasks, cybersecurity, reverse engineering, and low-level system code.
> 2. **Agentic Tool Calling & Subagent Orchestration:** Strict JSON function-calling support via Jinja templates; capable of delegating to and managing subagents without recursion loops.
> 3. **Massive Native Context:** At least 128k tokens (up to 262k or 512k tokens) functional with Flash Attention.
> 4. **MoE Preference for High Throughput:** Prefer Mixture of Experts (e.g., ~35B total with ~3B active, or linear attention architectures) capable of **70 to 130+ tok/s** on NVIDIA hardware. Avoid slow dense 70B models (~20 tok/s).
> 5. **VRAM constraints:**
>    - **L4 GPU (24GB VRAM):** GGUF file size <= 18 GB with 128k context.
>    - **A100 GPU (80GB VRAM):** GGUF file size <= 35-45 GB (Q6_K) with 262k context.
> 
> Please provide exact Hugging Face model IDs, GGUF quantizers (e.g. mradermacher, mudler, bartowski), and quantization recommendation."

---

## 🏷️ 4. Palabras Clave y Filtros Rápidos para Hugging Face

* **Términos de desensura:** `abliterated`, `uncensored`, `heretic`, `unaligned`, `no-refusal`.
* **Familias de modelos top:** `Qwen3.6`, `Qwen3.5`, `Claude-4.7-Opus-Distill`, `Kimi-Linear`, `Kimi-Dev`, `DeepSeek-V4`, `Hermes-4`.
* **Tags de arquitectura:** `moe`, `mixture-of-experts`, `function-calling`, `tool-use`, `gguf`.
