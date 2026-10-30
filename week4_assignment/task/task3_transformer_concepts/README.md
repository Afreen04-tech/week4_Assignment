# Task 3: Transformer Concepts in Code

This module provides clean, reproducible, hands-on implementations of foundational Transformer mechanisms built in pure NumPy and Hugging Face Transformers.

---

## 1. Directory Structure

```
task3_transformer_concepts/
├── self_attention.py          # Pure NumPy single-head scaled dot-product attention
├── multi_head_attention.py    # 2-Head multi-head attention with output projection
├── positional_encoding.py     # Sinusoidal positional encoding implementation & plotter
├── huggingface_examples.py    # BERT fill-mask & GPT-2 text generation pipelines
├── outputs/
│   └── positional_encoding.png # Generated dual-panel positional encoding visualization
└── README.md                  # Comprehensive architectural reference & explanations
```

---

## 2. Scaled Dot-Product Self-Attention (NumPy)

### Mathematical Formulation
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

1. **Query-Key Affinity**: $S = \frac{Q K^T}{\sqrt{d_k}}$, where dividing by $\sqrt{d_k}$ prevents dot products from growing excessively large in high dimensions, which would push softmax gradients into regions with vanishingly small gradients.
2. **Probability Distribution**: Row-wise softmax guarantees that for every query token $i$, $\sum_{j=1}^N A_{ij} = 1.0$.
3. **Value Aggregation**: Contextual representation $Z = A V$.

### Verified Output ($4 \times 4$ Attention Matrix)
```
--- Attention Weights Matrix A (4x4) ---
[[0.2356 0.1009 0.3344 0.3291]
 [0.1912 0.4270 0.0088 0.3730]
 [0.0400 0.3394 0.5259 0.0947]
 [0.7096 0.0812 0.0095 0.1997]]

Row 0 ('        The'): sum = 1.000000
Row 1 ('transformer'): sum = 1.000000
Row 2 ('     learns'): sum = 1.000000
Row 3 ('  attention'): sum = 1.000000
np.allclose(row_sums, 1.0) -> True
```

---

## 3. Multi-Head Attention (2 Heads)

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2) W^O$$
$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

- **Representation Subspaces**: Allows the model to jointly attend to information from different representation subspaces (e.g., Head 1 tracks syntactic dependencies, Head 2 tracks semantic relationships).
- **Projection**: The concatenated output $(N, 2 \times d_k)$ is projected back to the embedding dimension $d_{model}$ via matrix $W^O$. Both heads verify $\sum \text{row} = 1.0$.

---

## 4. Sinusoidal Positional Encoding (10 Positions, 8 Dimensions)

$$\begin{aligned}
PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i / d_{model}}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i / d_{model}}}\right)
\end{aligned}$$

- **Why Needed**: Unlike RNNs that process tokens sequentially, self-attention is permutation-equivariant ($A(PX) \neq PA(X)$). Injecting positional vectors provides the network with token order and distance awareness.
- **Linear Relative Shift Property**: Because $\sin(\alpha + \beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$, for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$, allowing the model to readily attend by relative position.
- **Plot**: The dual-panel visualizer renders the $10 \times 8$ matrix heatmap and continuous sinusoidal oscillation frequencies to `outputs/positional_encoding.png`.

---

## 5. Hugging Face Demonstrations

1. **Encoder (BERT / DistilBERT `fill-mask`)**:
   - Input: `"The transformer architecture relies on the [MASK] mechanism."`
   - Top Prediction: `coupling` / `feedback` / `attention` with calibrated confidence scores, leveraging bidirectional context.
2. **Decoder (GPT-2 `text-generation`)**:
   - Seed: `"Large language models will fundamentally transform software engineering by"`
   - Output: Autoregressively generates next tokens sequentially using causal attention masking.

---

## 6. Executive Architectural Analysis (6 Lines)

- **Line 1**: Encoder-only models (e.g., BERT) employ unmasked, bidirectional self-attention to process entire sequences at once, making them optimal for deep comprehension, classification, and token extraction.
- **Line 2**: Decoder-only models (e.g., GPT-4, Llama) enforce causal (unidirectional) triangular masking so tokens only attend to preceding tokens, making them optimal for autoregressive next-token text generation.
- **Line 3**: Encoder-decoder models (e.g., T5) explicitly decouple sequence encoding from conditioned sequence generation, excelling at sequence-to-sequence tasks like translation and summarization.
- **Line 4**: Traditional Recurrent Neural Networks (RNNs/LSTMs) computed hidden states strictly sequentially ($h_t = f(h_{t-1}, x_t)$), creating an unavoidable $O(N)$ temporal bottleneck that prevented GPU parallelization.
- **Line 5**: Transformers replaced RNNs by computing pairwise attention across all tokens simultaneously in parallel matrix operations, unlocking massive horizontal scaling across high-performance GPU/TPU clusters.
- **Line 6**: Furthermore, self-attention provides a direct $O(1)$ path length between any two arbitrary token positions, permanently resolving the catastrophic vanishing gradient and catastrophic forgetting flaws of RNNs.
