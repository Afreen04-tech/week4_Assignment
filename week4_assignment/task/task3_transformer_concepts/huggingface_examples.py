import os
import sys
import warnings

# Suppress non-critical library warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import pipeline


def run_bert_fill_mask():
    """
    Demonstrates bidirectional masked language modeling (Encoder architecture).
    Model predicts the masked token [MASK] using both left and right context.
    """
    print("\n" + "=" * 80)
    print("1. ENCODER DEMO: MASKED LANGUAGE MODELING (BERT / DistilBERT)")
    print("=" * 80)

    # Prefer distilbert-base-uncased or bert-base-uncased
    model_name = "distilbert/distilbert-base-uncased"
    print(f"Loading Encoder model: {model_name}...")

    unmasker = pipeline("fill-mask", model=model_name)

    test_sentence = "The transformer architecture relies on the [MASK] mechanism."
    print(f"\nInput Sentence: \"{test_sentence}\"")

    results = unmasker(test_sentence, top_k=5)

    print("\nTop 5 Mask Predictions (Bidirectional Context):")
    print(f"{'Rank':<5} | {'Predicted Token':<20} | {'Probability Score':<18} | {'Completed Sentence'}")
    print("-" * 80)
    for i, res in enumerate(results, start=1):
        token = res["token_str"].strip()
        score = res["score"]
        completed = res["sequence"]
        print(f"{i:<5} | {token:<20} | {score:18.4f} | \"{completed}\"")

    return results


def run_gpt2_text_generation():
    """
    Demonstrates causal, unidirectional autoregressive text generation (Decoder architecture).
    Model generates subsequent tokens one-by-one conditioned strictly on preceding tokens.
    """
    print("\n" + "=" * 80)
    print("2. DECODER DEMO: AUTOREGRESSIVE TEXT GENERATION (GPT-2)")
    print("=" * 80)

    model_name = "gpt2"
    print(f"Loading Decoder model: {model_name}...")

    generator = pipeline("text-generation", model=model_name)

    prompt = "Large language models will fundamentally transform software engineering by"
    print(f"\nSeed Prompt: \"{prompt}\"")

    generations = generator(
        prompt,
        max_new_tokens=40,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=50256
    )

    generated_text = generations[0]["generated_text"]
    print("\nGenerated Continuation:")
    print("-" * 80)
    print(generated_text)
    print("-" * 80)

    return generated_text


EXPLANATION_6_LINES = """
================================================================================
EXECUTIVE 6-LINE ANALYSIS: ENCODER VS DECODER & WHY TRANSFORMERS REPLACED RNNS
================================================================================
Line 1: Encoder-only models (e.g., BERT) employ unmasked, bidirectional self-attention to process entire sequences at once, making them optimal for deep comprehension, classification, and token extraction.
Line 2: Decoder-only models (e.g., GPT-4, Llama) enforce causal (unidirectional) triangular masking so tokens only attend to preceding tokens, making them optimal for autoregressive next-token text generation.
Line 3: Encoder-decoder models (e.g., T5) explicitly decouple sequence encoding from conditioned sequence generation, excelling at sequence-to-sequence tasks like translation and summarization.
Line 4: Traditional Recurrent Neural Networks (RNNs/LSTMs) computed hidden states strictly sequentially (h_t = f(h_{t-1}, x_t)), creating an unavoidable O(N) temporal bottleneck that prevented GPU parallelization.
Line 5: Transformers replaced RNNs by computing pairwise attention across all tokens simultaneously in parallel matrix operations, unlocking massive horizontal scaling across high-performance GPU/TPU clusters.
Line 6: Furthermore, self-attention provides a direct O(1) path length between any two arbitrary token positions, permanently resolving the catastrophic vanishing gradient and catastrophic forgetting flaws of RNNs.
================================================================================
"""


def main():
    print("=" * 80)
    print("TASK 3.4 & 3.5: HUGGING FACE DEMONSTRATIONS & ARCHITECTURAL ANALYSIS")
    print("=" * 80)

    # 1. BERT Fill-mask
    run_bert_fill_mask()

    # 2. GPT-2 Generation
    run_gpt2_text_generation()

    # 3. 6-Line Explanation
    print(EXPLANATION_6_LINES)


if __name__ == "__main__":
    main()
