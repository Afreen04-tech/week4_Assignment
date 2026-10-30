# Task 2: Capabilities and Limitations Lab

This directory contains empirical experiments evaluating the strengths, boundaries, and failure modes of Large Language Models (LLMs) when applied to enterprise workloads.

---

## 1. Directory Structure

```
task2_capabilities_limitations/
├── capabilities_lab.py          # Complete automated test runner and verification suite
├── sample_texts/
│   └── test_samples.json        # Test prompts, input texts, and ground truth data
├── results/
│   └── assessment_results.csv   # Comprehensive results table (Test | Prompt | Result | Rating)
└── README.md                    # Executive documentation and technical analysis
```

---

## 2. Test Coverage & Empirical Findings

| Test | Objective | Observed Result | Score (1-5) | Key Insight |
| :--- | :--- | :--- | :---: | :--- |
| **1. Summarisation** | Condense complex operational IT report | Extracted milestones, financial overruns, and security deadlines in 3 executive bullet points. | 5 / 5 | LLMs excel at synthesizing unstructured text into structured executive summaries. |
| **2. Classification** | Categorize support ticket & assign priority | Correctly identified root cause (`Technical`) over symptom (`Billing`) with `Critical` priority. | 5 / 5 | Strong semantic comprehension enables accurate multi-factor triage. |
| **3. Extraction** | Extract SLA terms into JSON | Extracted `contract_id`, `guaranteed_uptime_percent`, penalties, and email with exact datatypes. | 5 / 5 | Highly reliable for zero-shot schema extraction. |
| **4. Translation** | Translate technical infrastructure text to German & Spanish | Preserved specialized domain terms (`Kubernetes`, `Failover`, `WebSocket`, `Node-Upgrade`). | 5 / 5 | Cross-lingual domain transfer maintains technical fluency. |
| **5. Code Generation** | Generate Python `@retry_with_backoff` decorator | Implemented `functools.wraps`, exponential delay, typed exception catching, and logger warnings. | 5 / 5 | Generated code is syntactically sound, typed, and immediately executable. |
| **6. Knowledge Cutoff** | Query 2028 Olympic gold medalist | Transparently refused: recognized the date is beyond the temporal training cutoff. | 4 / 5 | Modern post-training instills awareness of temporal boundaries. |
| **7a. Hallucination (Unconstrained)** | Ask about a fabricated paper & researcher | Invented detailed fictional methodologies and 3 plausible-sounding quantum biology findings. | 1 / 5 | **Critical failure mode**: Unconstrained LLMs confabulate authoritatively when prompted with leading questions. |
| **7b. Hallucination (Safeguarded)** | Add explicit refusal instruction ("say I don't know") | Model successfully stated it does not know and identified that the paper does not exist. | 5 / 5 | Explicit negative constraints and calibrated refusal instructions curb hallucination dramatically. |
| **8. Context Length (Needle in Haystack)** | Retrieve token at 50% depth across 2,800-word log dump | Retrieved `OMEGA-DELTA-9472-RESTORE` with zero corruption. | 5 / 5 | Attention mechanisms maintain high fidelity for direct key-value lookup across long context. |
| **9. Multi-Step Arithmetic** | Calculate two-phase server pricing, discounts, and tiered data egress | Hand-verified calculation matched exactly ($3,392.996 -> $3,393.00). | 5 / 5 | Multi-step arithmetic succeeds when forced into chain-of-thought steps, but is vulnerable without code execution. |
| **10. Logic Puzzle** | Solve 4-chair seat ordering with negative constraints | Proved Diana=1, Bob=2, Charlie=3, Alice=4 through systematic deduction; answer: Charlie. | 5 / 5 | Constraint-satisfaction reasoning is reliable when step-by-step reasoning steps are articulated. |

---

## 3. Mathematical & Logical Hand Verification

### Arithmetic Problem Verification
- **Phase 1 Compute**: $14 \text{ servers} \times 48 \text{ hours} \times \$1.85/\text{hr} = \$1,243.20$
- **Phase 2 Compute**: $10 \text{ servers} \times 96 \text{ hours} \times \$2.60/\text{hr} = \$2,496.00$
- **Gross Compute**: $\$1,243.20 + \$2,496.00 = \$3,739.20$
- **Compute Discount (12%)**: $\$3,739.20 \times (1 - 0.12) = \$3,290.496$
- **Data Egress**:
  - First $1,000 \text{ GB} \times \$0.08/\text{GB} = \$80.00$
  - Remaining $450 \text{ GB} \times \$0.05/\text{GB} = \$22.50$
  - Total Egress = $\$102.50$ (no discount)
- **Final Hand-Verified Total**: $\$3,290.496 + \$102.50 = \mathbf{\$3,392.996}$ (or **\$3,393.00**)

### Logic Puzzle Verification
- **Chairs**: `[1, 2, 3, 4]`
- Diana is in 1: `[Diana, _, _, _]`. Remaining positions for Bob, Charlie, Alice are `[2, 3, 4]`.
- Charlie is immediately right of Bob: contiguous block `[Bob, Charlie]`.
  - If `[Bob, Charlie]` are in `[3, 4]`, then Alice must be in `2`. This places Alice right next to Bob, which violates the constraint.
  - If `[Bob, Charlie]` are in `[2, 3]`, then Alice must be in `4`. Here, Bob is in 2 and Alice is in 4 (separated by Charlie in 3). All constraints are satisfied.
- **Hand-Verified Answer**: **Charlie is sitting in chair 3**.

---

## 4. Five Essential Production Safeguards for Business Applications

1. **Deterministic Calculation Offloading**:
   Never rely on raw LLM token prediction for arithmetic, billing, tax, or financial calculations. Bind the LLM via tool-calling to a sandboxed Python runtime or microservice API where parameters are verified and executed with IEEE-754 precision.

2. **Calibrated Grounding & Epistemic Refusal (RAG)**:
   Mandate in the system prompt that the model must base answers exclusively on retrieved context documents. Instruct the model to explicitly state *"I cannot verify this from the provided documents"* whenever confidence falls below an empirical threshold.

3. **Dual-Layer Schema Validation with Pydantic**:
   Always couple structured output modes (JSON mode) with rigorous Pydantic schema validation. Enforce field types, regex patterns, enum bounds, and required keys to guarantee database safety.

4. **Bidirectional Safety Guardrails**:
   Deploy inbound guardrails to intercept prompt injections, jailbreaks, and system prompt leakage attacks. Deploy outbound guardrails to detect and redact Personally Identifiable Information (PII), API keys, and toxic content before reaching users.

5. **Human-in-the-Loop (HITL) Routing**:
   Establish automated confidence scoring and risk-tiered escalation. Any request involving financial refunds over threshold, destructive actions, or low semantic certainty must be staged in an administrative review queue for human verification.
