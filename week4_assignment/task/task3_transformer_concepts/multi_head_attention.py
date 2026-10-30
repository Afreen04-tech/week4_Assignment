

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention for a single head."""
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    attention_weights = softmax(scores, axis=-1)
    head_output = attention_weights @ V
    return head_output, attention_weights


class MultiHeadAttentionNumPy:
    

    def __init__(self, d_model: int = 8, num_heads: int = 2, seed: int = 42):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head: 8 // 2 = 4

        # Reproducible random projection matrices
        rng = np.random.default_rng(seed)

        # Projections for Head 1 and Head 2
        self.W_Q1 = rng.standard_normal((d_model, self.d_k)) * 0.5
        self.W_K1 = rng.standard_normal((d_model, self.d_k)) * 0.5
        self.W_V1 = rng.standard_normal((d_model, self.d_k)) * 0.5

        self.W_Q2 = rng.standard_normal((d_model, self.d_k)) * 0.5
        self.W_K2 = rng.standard_normal((d_model, self.d_k)) * 0.5
        self.W_V2 = rng.standard_normal((d_model, self.d_k)) * 0.5

        # Final output projection matrix W_O: (num_heads * d_k, d_model) -> (8, 8)
        self.W_O = rng.standard_normal((d_model, d_model)) * 0.5

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Forward pass for Multi-Head Attention.

        Args:
            X: Input embedding matrix of shape (seq_len, d_model)

        Returns:
            output: Multi-head contextual output of shape (seq_len, d_model)
            attention_maps: List of attention matrices [A1, A2], each of shape (seq_len, seq_len)
        """
        # Head 1 Projections & Attention
        Q1 = X @ self.W_Q1  # (seq_len, 4)
        K1 = X @ self.W_K1  # (seq_len, 4)
        V1 = X @ self.W_V1  # (seq_len, 4)
        head1_out, A1 = scaled_dot_product_attention(Q1, K1, V1)

        # Head 2 Projections & Attention
        Q2 = X @ self.W_Q2  # (seq_len, 4)
        K2 = X @ self.W_K2  # (seq_len, 4)
        V2 = X @ self.W_V2  # (seq_len, 4)
        head2_out, A2 = scaled_dot_product_attention(Q2, K2, V2)

        # Concatenate head outputs along feature dimension: (seq_len, 8)
        concat_heads = np.concatenate([head1_out, head2_out], axis=-1)

        # Apply output projection W_O: (seq_len, 8) @ (8, 8) -> (seq_len, 8)
        output = concat_heads @ self.W_O

        return output, [A1, A2]


def main():
    print("=" * 80)
    print("TASK 3.2: 2-HEAD MULTI-HEAD ATTENTION IMPLEMENTATION (NumPy)")
    print("=" * 80)

    seq_len = 4
    d_model = 8
    num_heads = 2
    tokens = ["The", "transformer", "learns", "attention"]

    np.random.seed(101)
    X = np.random.randn(seq_len, d_model)

    mha = MultiHeadAttentionNumPy(d_model=d_model, num_heads=num_heads, seed=42)
    output, attention_maps = mha.forward(X)

    print(f"Sequence Length: {seq_len} tokens")
    print(f"Embedding Dim (d_model): {d_model}")
    print(f"Number of Heads: {num_heads} (d_k per head = {d_model // num_heads})")

    for h, A in enumerate(attention_maps, start=1):
        print(f"\n--- Head {h} Attention Weights Matrix A_{h} (4x4) ---")
        print(np.round(A, 4))

        row_sums = np.sum(A, axis=-1)
        print(f"Head {h} Row Sums: {np.round(row_sums, 6)}")
        assert np.allclose(row_sums, 1.0), f"Head {h} rows do not sum to 1.0!"
        print(f"Head {h} Row-Wise Sum Verification: PASSED (all rows sum to 1.0)")

    print("\n--- Output Projection Multi-Head Tensor (seq_len, d_model) ---")
    print(np.round(output, 4))
    print(f"Final Output Tensor Shape: {output.shape} (matches input shape)")
    print("[SUCCESS] 2-Head Multi-Head Attention validated successfully.")


if __name__ == "__main__":
    main()
