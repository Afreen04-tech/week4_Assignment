

import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
PLOT_PATH = os.path.join(OUTPUT_DIR, "positional_encoding.png")


def compute_sinusoidal_positional_encoding(num_positions: int = 10, d_model: int = 8) -> np.ndarray:
    """
    Computes standard sinusoidal positional encodings.

    Args:
        num_positions: Sequence length (number of positions, e.g. 10)
        d_model: Dimensionality of embeddings (must be even, e.g. 8)

    Returns:
        PE: NumPy array of shape (num_positions, d_model)
    """
    assert d_model % 2 == 0, "d_model must be an even integer."

    pe = np.zeros((num_positions, d_model))
    positions = np.arange(num_positions)[:, np.newaxis]  # (num_positions, 1)

    # Compute division term: 10000^(2i / d_model)
    # Using log-space: exp(2i * -(log(10000.0) / d_model))
    i_indices = np.arange(0, d_model, 2)  # [0, 2, 4, 6]
    div_term = np.exp(i_indices * -(np.log(10000.0) / d_model))  # (d_model/2,)

    # Apply sin to even indices (2i) and cos to odd indices (2i+1)
    pe[:, 0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions * div_term)

    return pe


def plot_positional_encoding(pe: np.ndarray, save_path: str):
    """
    Generates a dual-panel visualization:
    1. Heatmap showing positional vectors across positions and dimensions.
    2. Line chart showing sinusoidal oscillation frequencies across dimensions.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Heatmap of Positional Encoding Matrix
    cax = axes[0].imshow(pe, cmap="viridis", aspect="auto")
    axes[0].set_title("Sinusoidal Positional Encoding Matrix (10x8)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Embedding Dimension (d = 0..7)", fontsize=10)
    axes[0].set_ylabel("Token Position (pos = 0..9)", fontsize=10)
    axes[0].set_xticks(range(pe.shape[1]))
    axes[0].set_yticks(range(pe.shape[0]))
    fig.colorbar(cax, ax=axes[0], label="Encoding Value")

    # Annotate values inside heatmap cells
    for i in range(pe.shape[0]):
        for j in range(pe.shape[1]):
            val = pe[i, j]
            text_color = "white" if abs(val) < 0.5 else "black"
            axes[0].text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    # Panel 2: Continuous sinusoidal wave patterns for selected dimensions
    pos_axis = np.linspace(0, pe.shape[0] - 1, 150)
    d_model = pe.shape[1]
    for d in range(0, d_model, 2):
        freq = np.exp(d * -(np.log(10000.0) / d_model))
        axes[1].plot(pos_axis, np.sin(pos_axis * freq), label=f"Dim {d} (sin, high freq)" if d == 0 else f"Dim {d} (sin)")
        axes[1].plot(pos_axis, np.cos(pos_axis * freq), linestyle="--", label=f"Dim {d+1} (cos)")

    axes[1].set_title("Frequency Progression Across Dimensions", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Position in Sequence", fontsize=10)
    axes[1].set_ylabel("Value", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved positional encoding plot to: {save_path}")


def main():
    print("=" * 80)
    print("TASK 3.3: SINUSOIDAL POSITIONAL ENCODING (10 Positions, 8 Dimensions)")
    print("=" * 80)

    num_positions = 10
    d_model = 8

    pe = compute_sinusoidal_positional_encoding(num_positions, d_model)

    print("\n--- Positional Encoding Matrix (10 x 8) ---")
    header = "Pos | " + " | ".join([f"Dim {i}" for i in range(d_model)])
    print(header)
    print("-" * len(header))
    for p in range(num_positions):
        row_str = f" {p:>2} | " + " | ".join([f"{pe[p, d]:6.3f}" for d in range(d_model)])
        print(row_str)

    # Dot product similarity between position 0 and subsequent positions
    print("\n--- Positional Vector Orthogonality & Dot Product Similarity (pos=0 vs others) ---")
    for p in range(num_positions):
        sim = np.dot(pe[0], pe[p]) / (np.linalg.norm(pe[0]) * np.linalg.norm(pe[p]))
        print(f"Cosine Similarity(PE[0], PE[{p}]): {sim:+.4f}")

    plot_positional_encoding(pe, PLOT_PATH)


if __name__ == "__main__":
    main()
