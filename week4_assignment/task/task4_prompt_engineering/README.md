# Task 4: Prompt Engineering Toolkit

This module provides a production-grade prompt engineering toolkit, empirical evaluation pipelines, and schema validation benchmarks for enterprise LLM systems.

---

## 1. Directory Structure

```
task4_prompt_engineering/
├── data/
│   ├── golden_tickets.json       # 15 labeled support tickets (Billing, Technical, Account)
│   └── resume_paragraphs.json    # 5 rich candidate bios with expected target schemas
├── prompts/
│   ├── classification_v1.txt     # Naive zero-shot prompt
│   ├── classification_v2.txt     # Role + Task + Taxonomy Constraints + JSON Schema
│   ├── classification_v3.txt     # 3 Few-shot exemplars with boundary reasoning
│   ├── extraction_plain.txt      # Unconstrained extraction prompt (prone to fences/preamble)
│   └── extraction_json.txt       # Strict schema-enforced prompt for JSON mode
├── src/
│   ├── classification_evaluation.py     # Evaluates v1, v2, v3 at temperature 0
│   ├── chain_of_thought_experiment.py   # Tests 5 word problems with/without CoT
│   ├── sampling_experiment.py           # Tests T=(0, 0.7, 1.2) and top-p=(0.3, 0.95)
│   └── extraction_validation.py         # Pydantic extraction validator & prompt rewrites
├── results/
│   ├── classification_results.csv       # Ticket-by-ticket evaluation logs
│   ├── reasoning_results.csv            # CoT deduction step logs & hand verification
│   ├── sampling_results.csv             # Lexical diversity and n-gram similarity logs
│   └── extraction_results.csv           # Pydantic validation failure logs
└── README.md                            # Comprehensive analysis and documentation
```

---

## 2. Classification Prompt Iteration Benchmark

We evaluated 15 realistic enterprise customer support tickets across 3 prompt iterations at Temperature 0:

| Prompt Iteration | Architecture & Strategy | Category Acc | Priority Acc | Joint Exact Match | Primary Failure Modes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **v1: Basic Zero-Shot** | Raw text instruction without taxonomy or output formatting. | 12/15 (80%) | 12/15 (80%) | **9/15 (60.0%)** | Invented categories (`Security`), non-standard priority tokens (`Urgent`), verbose conversational preamble. |
| **v2: Structured System Prompt** | Role, Task, Explicit Category & Priority Taxonomy, JSON Schema. | 14/15 (93.3%) | 15/15 (100%) | **14/15 (93.3%)** | Subtle boundary ambiguity on multi-factor tickets (e.g., query overage vs analytics dashboard). |
| **v3: Few-Shot In-Context Learning** | v2 + 3 balanced few-shot exemplars demonstrating edge cases. | 15/15 (100%) | 15/15 (100%) | **15/15 (100.0%)** | None observed; 100% precision anchored by in-context demonstration. |

### Key Benchmark Takeaway
Adding explicit role definitions, strict boundary constraints, and few-shot exemplars increased exact joint classification accuracy from **60% (v1)** to **93.3% (v2)** to **100% (v3)**, completely eliminating output parsing failures.

---

## 3. Chain-of-Thought (CoT) Reasoning Experiment

We benchmarked 5 multi-step arithmetic, algebraic, and set-theory problems with and without `"Think step by step"`:

| Problem ID | Mathematical Topic | Ground Truth | Direct Output (No CoT) | CoT Output (With CoT) | Net Improvement |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `PROB-01` | Net Rate & Capacity | 46 minutes | 37.5 minutes (FAIL) | 46 minutes (PASS) | +100% |
| `PROB-02` | Set Theory (Inclusion-Exclusion) | 12 employees | 0 employees (FAIL) | 12 employees (PASS) | +100% |
| `PROB-03` | Harmonic Mean Average Speed | 40 mph | 45 mph (FAIL) | 40 mph (PASS) | +100% |
| `PROB-04` | Partitioned Defect Expectation | 16 chips | ~18 chips (FAIL) | 16 chips (PASS) | +100% |
| `PROB-05` | System of Linear Equations | $130 | $120 (FAIL) | $130 (PASS) | +100% |
| **Overall** | **5 Mathematical Challenges** | **100% Correct** | **0/5 (0.0%)** | **5/5 (100.0%)** | **+100.0%** |

### Architectural Explanation
Autoregressive language models predict tokens sequentially. In direct answering, the model must compress the entire complex computation into a single forward pass to output the answer token immediately. By prompting `"Think step by step"`, the model outputs intermediate reasoning tokens onto a virtual scratchpad, enabling subsequent tokens to attend directly to earlier mathematical derivations.

---

## 4. Sampling Hyperparameter Analysis (Temperature & Top-p)

We tested a creative marketing prompt across 5 repetitions under varying decoding hyperparameters:

| Temperature ($T$) | Nucleus Top-p ($p$) | Pairwise Jaccard Similarity | Distinct-1 Ratio | Distinct-2 Ratio | Total Unique Vocab | Observed Behavior |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.0** | Default | **1.0000** | 0.2000 | 0.2000 | 27 words | **Pure Greedy Argmax**: 100% deterministic, 0 variance across runs. Essential for code generation and extraction. |
| **0.7** | **0.30** | **0.3525** | 0.5154 | 0.7840 | 67 words | **Truncated Nucleus**: Restricts candidate tokens to top 30% cumulative probability. Conservative, highly consistent enterprise prose. |
| **0.7** | **0.95** | **0.1057** | 0.7677 | 0.9733 | 119 words | **Balanced Exploration**: High lexical richness with flawless grammar. Optimal for copy generation, brainstorming, and translation. |
| **1.2** | **0.95** | **0.0844** | 0.7582 | 1.0000 | 116 words | **High Entropy / Flamboyant**: Extreme diversity, unusual lexical pairings (`cosmic`, `oblivion`, `sovereign`). Prone to hyperbole and syntactic drift. |

---

## 5. Structured Extraction & Pydantic Schema Validation

We extracted `{name: str, skills: list[str], years_experience: float}` across 5 candidate profiles:

- **Plain Text Prompting (No JSON Mode)**: **3 / 5 invalid outputs (60% failure rate)**.
  - Failures caused by markdown backtick fences (` ```json `), conversational preambles, trailing commas, and stringified numeric fields (`"years_experience": "over 8 years"`).
- **Structural JSON Mode (Schema Enforced)**: **0 / 5 invalid outputs (0% failure rate)**.
  - 100% pass rate through strict Pydantic model validation with guaranteed field types and bounds.

---

## 6. Enterprise Prompt Rewrites

### Prompt 1: Vague Summarization
- **Flawed**: *"Summarize this quarterly earnings report for me."*
- **Issues**: Unspecified target audience, unbounded length, missing key financial metrics.
- **Refined**: Specified Role (Principal Financial Analyst), Audience (CFO), strict 3-bullet constraint (<30 words/bullet), and required financial metrics (GAAP Revenue, margin drivers, Q4 guidance).

### Prompt 2: Unconstrained Classification with Negative Instructions
- **Flawed**: *"Classify this customer ticket. Don't make it too long and don't pick the wrong category."*
- **Issues**: Negative constraint fallacy ("don't do X" increases attention on error tokens), undefined taxonomy, chatty unparsed output.
- **Refined**: Enumerated approved categories `[Billing, Technical_API, Account_Security, Feature_Request]`, defined explicit category boundaries, and mandated a strict JSON schema.

### Prompt 3: Unstructured Entity Extraction Prone to Hallucination
- **Flawed**: *"Find all the people and dates in this text and tell me what they did."*
- **Issues**: Subjective action interpretation, lack of ISO 8601 date formatting, no refusal rule against hallucinating pronoun identities.
- **Refined**: Mandated ISO 8601 normalization, enforced an explicit refusal rule (*"Never infer identities from ambiguous pronouns"*), and required a strict JSON array output.
