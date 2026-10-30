
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable row-wise softmax.
    Subtracts row maximums to prevent exp overflow.
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes Scaled Dot-Product Self-Attention.

    Args:
        Q: Queries matrix of shape (seq_len, d_k)
        K: Keys matrix of shape (seq_len, d_k)
        V: Values matrix of shape (seq_len, d_v)

    Returns:
        output: Contextual output matrix of shape (seq_len, d_v)
        attention_weights: Normalized attention probability matrix of shape (seq_len, seq_len)
    """
    d_k = Q.shape[-1]

    # 1. Compute raw affinity scores: (seq_len, seq_len)
    scores = (Q @ K.T) / np.sqrt(d_k)

    # 2. Apply row-wise softmax to obtain attention distribution
    attention_weights = softmax(scores, axis=-1)

    # 3. Weighted aggregation of values: (seq_len, d_v)
    output = attention_weights @ V

    return output, attention_weights


def main():
    print("=" * 80)
    print("TASK 3.1: SINGLE-HEAD SELF-ATTENTION IMPLEMENTATION (NumPy)")
    print("=" * 80)

    # Deterministic seed for reproducible evaluation
    np.random.seed(42)

    seq_len = 4   # 4 tokens: e.g. ["The", "transformer", "learns", "attention"]
    d_model = 8   # Input embedding dimension
    d_k = 4       # Query/Key dimension
    d_v = 4       # Value dimension

    tokens = ["The", "transformer", "learns", "attention"]
    print(f"Tokens in sequence (N={seq_len}): {tokens}")

    # Simulated input embeddings X: (seq_len, d_model)
    X = np.random.randn(seq_len, d_model)

    # Learned projection weight matrices
    W_Q = np.random.randn(d_model, d_k) * 0.5
    W_K = np.random.randn(d_model, d_k) * 0.5
    W_V = np.random.randn(d_model, d_v) * 0.5

    # Linear projections to Query, Key, and Value spaces
    Q = X @ W_Q  # (4, 4)
    K = X @ W_K  # (4, 4)
    V = X @ W_V  # (4, 4)

    output, A = scaled_dot_product_attention(Q, K, V)

    print("\n--- 1. Query Matrix Q (4x4) ---")
    print(np.round(Q, 4))

    print("\n--- 2. Key Matrix K (4x4) ---")
    print(np.round(K, 4))

    print("\n--- 3. Value Matrix V (4x4) ---")
    print(np.round(V, 4))

    print("\n--- 4. Attention Weights Matrix A = softmax(Q @ K.T / sqrt(d_k)) (4x4) ---")
    print(np.round(A, 4))

    # Row-wise sum verification
    row_sums = np.sum(A, axis=-1)
    print("\n--- 5. Row-Wise Sum Verification (Must equal 1.0) ---")
    for i, (tok, s) in enumerate(zip(tokens, row_sums)):
        print(f"Row {i} (Token '{tok:>11}'): sum = {s:.6f}")

    is_valid = np.allclose(row_sums, 1.0)
    print(f"\nVerification Check: np.allclose(row_sums, 1.0) -> {is_valid}")
    assert is_valid, "Attention matrix rows do not sum to 1.0!"

    print("\n--- 6. Contextual Output Matrix (A @ V) (4x4) ---")
    print(np.round(output, 4))
    print(f"Output Tensor Shape: {output.shape}")
    print("[SUCCESS] Single-head self-attention validated successfully.")


if __name__ == "__main__":
    main()
