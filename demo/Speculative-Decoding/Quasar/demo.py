from __future__ import annotations

import numpy as np


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def fake_lowbit_quantize(w: np.ndarray, bits: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple symmetric per-row quantization.
    Educational only.
    """
    scale = np.max(np.abs(w), axis=1, keepdims=True) + 1e-8
    qmax = 2 ** (bits - 1) - 1
    q = np.round(np.clip(w / scale, -1.0, 1.0) * qmax).astype(np.int8)
    return q, scale


def fake_lowbit_dequantize(q: np.ndarray, scale: np.ndarray, bits: int = 8) -> np.ndarray:
    qmax = 2 ** (bits - 1) - 1
    return q.astype(np.float32) / qmax * scale


class TinyTargetModel:
    """
    Tiny target model used both for full-precision verification
    and quantized verification.
    """

    def __init__(self, hidden_dim: int, vocab_size: int):
        self.W = np.random.randn(
            hidden_dim, vocab_size).astype(np.float32) * 0.2

    def logits(self, hidden: np.ndarray) -> np.ndarray:
        return hidden @ self.W

    def quantized_logits(self, hidden: np.ndarray, bits: int = 8) -> np.ndarray:
        # quantize per output row
        q, s = fake_lowbit_quantize(self.W.T, bits=bits)
        Wq = fake_lowbit_dequantize(q, s, bits=bits).T
        return hidden @ Wq


def make_hidden_states(batch: int, hidden_dim: int) -> np.ndarray:
    return np.random.randn(batch, hidden_dim).astype(np.float32)


def make_draft_tokens(logits: np.ndarray) -> np.ndarray:
    """
    Pretend a draft path proposed these tokens.
    We use argmax from the full model with some random perturbation to mimic drafted candidates.
    """
    pred = np.argmax(logits, axis=-1)
    noise_mask = np.random.rand(*pred.shape) < 0.15
    pred = pred.copy()
    pred[noise_mask] = np.random.randint(
        0, logits.shape[-1], size=int(np.sum(noise_mask)))
    return pred


def acceptance_rate(full_logits: np.ndarray, verify_logits: np.ndarray, draft_tokens: np.ndarray) -> float:
    full_accept = np.argmax(full_logits, axis=-1) == draft_tokens
    verify_accept = np.argmax(verify_logits, axis=-1) == draft_tokens
    # compare verification decisions with full-precision decisions
    return float(np.mean(full_accept == verify_accept))


def logit_mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def estimated_weight_bytes_full(model: TinyTargetModel) -> int:
    return model.W.nbytes


def estimated_weight_bytes_quantized(model: TinyTargetModel, bits: int = 8) -> int:
    # toy estimate: int8 weights + fp32 scale per output row
    vocab_size = model.W.shape[1]
    hidden_dim = model.W.shape[0]
    weight_bytes = hidden_dim * vocab_size * (bits / 8.0)
    scale_bytes = vocab_size * 4
    return int(weight_bytes + scale_bytes)


def main() -> None:
    set_seed(0)

    batch = 512
    hidden_dim = 256
    vocab_size = 128

    hidden = make_hidden_states(batch, hidden_dim)
    model = TinyTargetModel(hidden_dim, vocab_size)

    full_logits = model.logits(hidden)
    draft_tokens = make_draft_tokens(full_logits)

    q8_logits = model.quantized_logits(hidden, bits=8)
    q4_logits = model.quantized_logits(hidden, bits=4)

    print("=== Quasar-inspired verification demo ===\n")
    print(f"hidden shape: {hidden.shape}")
    print(f"vocab size  : {vocab_size}\n")

    print("Logit fidelity")
    print(f"  full vs q8 MSE : {logit_mse(full_logits, q8_logits):.6f}")
    print(f"  full vs q4 MSE : {logit_mse(full_logits, q4_logits):.6f}\n")

    print("Verification-decision agreement with full precision")
    print(
        f"  q8 agreement   : {acceptance_rate(full_logits, q8_logits, draft_tokens):.4f}")
    print(
        f"  q4 agreement   : {acceptance_rate(full_logits, q4_logits, draft_tokens):.4f}\n")

    full_bytes = estimated_weight_bytes_full(model)
    q8_bytes = estimated_weight_bytes_quantized(model, bits=8)
    q4_bytes = estimated_weight_bytes_quantized(model, bits=4)

    print("Verification-stage weight traffic proxy")
    print(f"  full precision : {full_bytes / 1024:.1f} KB")
    print(f"  q8 estimate    : {q8_bytes / 1024:.1f} KB")
    print(f"  q4 estimate    : {q4_bytes / 1024:.1f} KB")
    print()

    print("Interpretation:")
    print("- Verification can be approximated with quantized weights while staying close to full logits.")
    print("- Lower precision reduces the memory traffic proxy for the verification pass.")
    print("- Real Quasar is more sophisticated; this demo only illustrates the central intuition.")


if __name__ == "__main__":
    main()
