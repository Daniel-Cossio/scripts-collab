# 📋 Colab Runbook: Snippets de Ejecución Inmediata

Guía rápida de bloques de código para copiar y pegar directamente en una celda de [Google Colab](https://colab.research.google.com/). Cada celda descarga limpiamente el repositorio y arranca el entorno sin riesgo de errores de indentación o saltos de línea.

---

## ⚡ 1. Auto-Detección Inteligente (Recomendado - Cualquier GPU)
Detecta si Colab te asignó Tesla T4, NVIDIA L4, A100 40GB o A100 80GB y configura automáticamente el Tier y contexto óptimo:

```python
# ==============================================================================
# COLAB LAUNCHER - AUTO-DETECCIÓN INTELIGENTE
# ==============================================================================
!rm -rf /content/scripts-collab
!git clone https://github.com/Daniel-Cossio/scripts-collab.git /content/scripts-collab
%run /content/scripts-collab/colab_server_adaptive.py
```

---

## 🚀 2. Tier 4: NVIDIA A100 80GB (MoE Frontier 262k Contexto - 130 tok/s)
Arranca `Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-APEX` en cuantización de alta fidelidad `Q6_K` con **262,144 tokens** de contexto:

```python
# ==============================================================================
# COLAB LAUNCHER - TIER 4: NVIDIA A100 (80 GB VRAM)
# Modelo: Huihui Qwen 3.6 35B-A3B Claude 4.7 Opus APEX (MoE 130 tok/s - 262k ctx)
# ==============================================================================
!rm -rf /content/scripts-collab
!git clone https://github.com/Daniel-Cossio/scripts-collab.git /content/scripts-collab
%run /content/scripts-collab/colab_tier4_a100_80gb.py
```

> [!TIP]
> **Modelo alternativo denso 72B:** Para evaluar `Kimi-Dev-72B-abliterated` (128k ctx, 22 tok/s):
> ```python
> import os
> os.environ["COLAB_MODEL_TIER4"] = "2"
> !rm -rf /content/scripts-collab
> !git clone https://github.com/Daniel-Cossio/scripts-collab.git /content/scripts-collab
> %run /content/scripts-collab/colab_tier4_a100_80gb.py
> ```

---

## 🌙 3. Tier 3: NVIDIA A100 40GB (MoE + Atención Lineal - 512k Contexto)
Arranca `Kimi-Linear-48B-A3B-Instruct-abliterated` en cuantización `i1-Q4_K_M` (29.7 GB) con **524,288 tokens** de contexto nativo:

```python
# ==============================================================================
# COLAB LAUNCHER - TIER 3: NVIDIA A100 (40 GB VRAM)
# Modelo: Kimi Linear 48B-A3B Abliterated (MoE 3B activos, 512k ctx, kv: f16)
# ==============================================================================
!rm -rf /content/scripts-collab
!git clone https://github.com/Daniel-Cossio/scripts-collab.git /content/scripts-collab
%run /content/scripts-collab/colab_tier3_a100.py
```

---

## ⚡ 4. Tier 2: NVIDIA L4 24GB (Colab Pro - SWE Profesional MoE 128k)
Arranca `Huihui-Qwen3.6-35B-A3B-abliterated` en cuantización `Q3_K_M` con **131,072 tokens** de contexto a **~90 tok/s**:

```python
# ==============================================================================
# COLAB LAUNCHER - TIER 2: NVIDIA L4 (24 GB VRAM)
# Modelo: Huihui Qwen 3.6 35B-A3B MoE (128k ctx, ~90 tok/s)
# ==============================================================================
!rm -rf /content/scripts-collab
!git clone https://github.com/Daniel-Cossio/scripts-collab.git /content/scripts-collab
%run /content/scripts-collab/colab_tier2_l4.py
```

---

## 🧠 5. Tier 1: Tesla T4 15GB (Entorno Gratuito - 128k Contexto)
Arranca `DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED` en cuantización `i1-Q4_K_M` con **131,072 tokens** de contexto a **45 tok/s** sin gastar créditos de cómputo:

```python
# ==============================================================================
# COLAB LAUNCHER - TIER 1: TESLA T4 (15.3 GB VRAM - GRATIS)
# Modelo: DeepSeek-V4-Pro Qwen 3.5 9B Uncensored (128k ctx, 45 tok/s)
# ==============================================================================
!rm -rf /content/scripts-collab
!git clone https://github.com/Daniel-Cossio/scripts-collab.git /content/scripts-collab
%run /content/scripts-collab/colab_tier1_t4.py
```

---

## 🔗 6. Conexión en Terminal Local (CachyOS / KDE Konsole)

Cuando el script en Colab imprima la URL de Cloudflare:

```bash
# 1. Conectar túnel al puerto local 8898
colab-connect https://<SUBDOMINIO-CLOUDFLARE>.trycloudflare.com

# 2. Iniciar agente OpenCode
opencode
```
