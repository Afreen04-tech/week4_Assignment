# Week 4 - Weekly Assessment: LLM Fundamentals

Comprehensive, individual, hands-on engineering lab covering foundation models, tokenization, cost economics, capabilities & limitations, attention mechanisms, positional encoding, and enterprise prompt engineering.

---

## 1. Project Overview & Scenario

An enterprise organization seeking to integrate Large Language Models (LLMs) into their product suite requires empirical benchmarks on model capabilities, operational token costs, failure modes, mathematical transformer foundations, and production prompt engineering.

This repository implements all 4 assessment tasks with runnable code, reproducible evaluations, verified mathematical hand proofs, and structured outputs.

---

## 2. Directory Structure

```
E:\task\
│
├── task1_models_tokens_cost.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── task2_capabilities_limitations\
│   ├── capabilities_lab.py
│   ├── sample_texts\
│   │   └── test_samples.json
│   ├── results\
│   │   └── assessment_results.csv
│   └── README.md
│
├── task3_transformer_concepts\
│   ├── self_attention.py
│   ├── multi_head_attention.py
│   ├── positional_encoding.py
│   ├── huggingface_examples.py
│   ├── outputs\
│   │   └── positional_encoding.png
│   └── README.md
│
├── task4_prompt_engineering\
│   ├── data\
│   │   ├── golden_tickets.json
│   │   └── resume_paragraphs.json
│   ├── prompts\
│   │   ├── classification_v1.txt
│   │   ├── classification_v2.txt
│   │   ├── classification_v3.txt
│   │   ├── extraction_plain.txt
│   │   └── extraction_json.txt
│   ├── src\
│   │   ├── classification_evaluation.py
│   │   ├── chain_of_thought_experiment.py
│   │   ├── sampling_experiment.py
│   │   └── extraction_validation.py
│   ├── results\
│   │   ├── classification_results.csv
│   │   ├── reasoning_results.csv
│   │   ├── sampling_results.csv
│   │   └── extraction_results.csv
│   └── README.md
│
└── .env.example
```

---

## 3. Quickstart & Execution Guide

### Prerequisites
- Python 3.10+ (Verified on Python 3.14)
- Git

### Installation
```bash
# 1. Clone repository or navigate to workspace
cd E:\task

# 2. Install dependencies
pip install -r requirements.txt
```

### Running All Tasks

```bash
# Task 1: Models, Tokens and Cost Assessment
python task1_models_tokens_cost.py

# Task 2: Capabilities and Limitations Lab
python task2_capabilities_limitations/capabilities_lab.py

# Task 3: Transformer Concepts in Code
python task3_transformer_concepts/self_attention.py
python task3_transformer_concepts/multi_head_attention.py
python task3_transformer_concepts/positional_encoding.py
python task3_transformer_concepts/huggingface_examples.py

# Task 4: Prompt Engineering Toolkit
python task4_prompt_engineering/src/classification_evaluation.py
python task4_prompt_engineering/src/chain_of_thought_experiment.py
python task4_prompt_engineering/src/sampling_experiment.py
python task4_prompt_engineering/src/extraction_validation.py
```

---

## 4. Summary of Tasks & Key Findings

### Task 1: Models, Tokens and Cost
- **Model Families Comparison**: Detailed analysis across GPT (OpenAI), Claude (Anthropic), Llama (Meta), and Gemini (Google DeepMind) with official citations.
- **Token Counting with `tiktoken`**:
  - English text demonstrates ~0.75-0.85 words/token (~1.2-1.3 tokens/word).
  - Indic languages (Tamil/Hindi) suffer from byte-fallback tokenization fragmentation (~0.15-0.3 words/token, or 3.5-6.5 tokens/word), driving token costs 2x to 3x higher.
- **Monthly Cost Calculation (`estimate_cost`)**:
  - Scenario: 1,000 chats/day $\times$ 30 days = 30,000 chats/month (45M input tokens, 12M output tokens).
  - **GPT-4o-mini**: **\$13.95/month** | **Gemini 1.5 Flash**: **\$6.97/month**
  - **GPT-4o**: **\$232.50/month** | **Claude 3.5 Sonnet**: **\$315.00/month**
  - **Llama 3.1 70B (On-Premises)**: **Fixed Hardware / CapEx** ($0.00 marginal vendor API cost).
- **Context Capacity**: Evaluated 8k vs 128k context limits against support threads, swagger specs, employee handbooks, and codebases.
- **Enterprise Recommendations**:
  - *Public Customer-Support Bot*: GPT-4o-mini (sub-500ms latency, \$13.95/mo, 99.9% managed SLA, structured JSON mode).
  - *Private On-Premise Tool*: Llama 3.1 / 3.3 70B (strict air-gapped data sovereignty, zero data exfiltration risk, HIPAA/GDPR compliance, custom fine-tuning).

### Task 2: Capabilities and Limitations Lab
- **Core Capabilities (Rated 1-5)**:
  - Summarisation: 5/5 | Classification: 5/5 | Extraction: 5/5 | Translation: 5/5 | Code Generation: 5/5.
- **Knowledge Cutoff**: Correctly acknowledged future temporal boundary for 2028 Olympic inquiries.
- **Hallucination Dynamics**:
  - *Unconstrained*: Severe hallucination (Score 1/5) inventing fictional papers and quantum neuroscience findings.
  - *With Safeguard*: Honest epistemic refusal (Score 5/5) when instructed *"say I don't know if unsure"*.
- **Needle In A Haystack**: 100% accurate retrieval of `OMEGA-DELTA-9472-RESTORE` buried at 50% depth across a 2,800-word log dump.
- **Hand-Verified Reasoning**:
  - Arithmetic Problem: Hand-verified total = **\$3,392.996** (or **\$3,393.00**).
  - Logic Puzzle: Unique valid assignment proved (Diana=1, Bob=2, Charlie=3, Alice=4); **Chair 3 is Charlie**.
- **5 Production Safeguards**: Deterministic code offloading, calibrated RAG refusal, dual-layer Pydantic validation, bidirectional safety guardrails, and Human-in-the-Loop (HITL) escalation.

### Task 3: Transformer Concepts in Code
- **Single-Head Self-Attention**: Pure NumPy implementation of $Q, K, V$, scaled scores $Q K^T / \sqrt{d_k}$, row-wise softmax, and output matrix. Verified all row sums equal **1.000000** (`np.allclose(row_sums, 1.0) == True`).
- **2-Head Multi-Head Attention**: Implemented parallel head projections, scaled dot-product attention, feature concatenation, and output projection matrix $W_O$. Verified both heads sum to 1.0.
- **Sinusoidal Positional Encoding**: Generated $10 \times 8$ matrix and plotted dual-panel visualizer (heatmap + wave frequencies) to `outputs/positional_encoding.png`.
- **Hugging Face Pipelines**:
  - BERT (`fill-mask`): Predicted masked tokens with calibrated probabilities using bidirectional context.
  - GPT-2 (`text-generation`): Autoregressively generated completions using causal triangular masking.
- **Architectural Analysis (6 Lines)**: Explained encoder-only vs decoder-only architectures and why transformers replaced RNNs ($O(N)$ sequential bottleneck eliminated; $O(1)$ constant attention path length).

### Task 4: Prompt Engineering Toolkit
- **Golden Test Sets**: 15 customer support tickets and 5 candidate resume bios with target JSON schemas.
- **Prompt Iteration Benchmark (Temperature 0)**:
  - **v1 (Basic Zero-Shot)**: Exact Match = **9/15 (60.0%)** (format drift, invented classes, priority hallucinations).
  - **v2 (Role + Constraints + Schema)**: Exact Match = **14/15 (93.3%)** (eliminated format errors).
  - **v3 (Few-Shot Exemplars + CoT)**: Exact Match = **15/15 (100.0%)** (perfect boundary disambiguation).
- **Chain-of-Thought (CoT)**:
  - Without CoT (Direct): **0/5 (0.0%)** on multi-step arithmetic, harmonic speeds, and set overlaps.
  - With CoT ("Think step by step"): **5/5 (100.0%)** (+100% accuracy gain).
- **Sampling Hyperparameters**:
  - $T=0.0$: Jaccard = 1.000 (100% identical greedy argmax).
  - $T=0.7$, Top-$p=0.95$: Jaccard = 0.106 (Optimal balanced diversity and coherence).
  - $T=1.2$, Top-$p=0.95$: Jaccard = 0.084 (Flamboyant entropy, extreme vocabulary, syntactic risk).
  - Top-$p=0.3$ vs $0.95$: Nucleus truncation restricts decoding to conservative, high-probability tokens.
- **Structured Extraction & Pydantic Validation**:
  - Plain Text: **3 / 5 invalid outputs (60% failure rate)**.
  - JSON Mode: **0 / 5 invalid outputs (0% failure rate)**.
- **Prompt Rewrites**: Complete before/after refactoring of 3 flawed enterprise prompts.

---

## 5. Effective Prompt in Cursor

When working with LLM workflows and prompt engineering inside Cursor or similar AI-assisted IDEs, use the following structured system prompt to ensure optimal code generation, schema conformance, and architectural consistency:

```markdown
You are an expert Principal AI Systems Engineer and LLM Architect.

When completing tasks in this repository, strictly adhere to these principles:
1. Deterministic Evaluation First: Always verify mathematical and algorithmic logic with automated tests and hand-verified assertions before asserting correctness.
2. Structured Schema Enforcement: Never emit raw, unparsed JSON or conversational preambles when structured data is requested; use Pydantic models with explicit field constraints.
3. Temperature & Decoding Hygiene: Use Temperature 0 for deterministic classification, code generation, and extraction. Reserve Temperature 0.7-1.0 and Top-p 0.9-0.95 strictly for creative synthesis.
4. Epistemic Calibration: If an inquiry references events beyond training cutoff dates or fabricated sources, explicitly state "I don't know" rather than confabulating.
5. Transformer Grounding: Ground all transformer explanations in the formal mathematical mechanics of Vaswani et al. (Scaled dot-product attention, causal masking, and sinusoidal frequency progressions).
```
