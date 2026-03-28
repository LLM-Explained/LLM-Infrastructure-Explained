from __future__ import annotations

import numpy as np


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


# -----------------------------
# TurboQuant-like toy codec
# -----------------------------
class TurboQuantToy:
    """
    Educational toy approximation:
    1) random orthogonal rotation
    2) uniform quantization to low bit-width
    3) 1-bit residual sign correction

    This is NOT the real TurboQuant algorithm.
    It only captures the broad systems intuition.
    """

    def __init__(self, dim: int, bits: int = 3):
        self.dim = dim
        self.bits = bits
        self.R = self._random_orthogonal(dim)

    @staticmethod
    def _random_orthogonal(dim: int) -> np.ndarray:
        q, _ = np.linalg.qr(np.random.randn(dim, dim))
        return q

    def compress(self, x: np.ndarray) -> dict:
        y = x @ self.R
        max_abs = np.max(np.abs(y), axis=1, keepdims=True) + 1e-8

        levels = 2 ** self.bits
        qmax = levels // 2 - 1
        q = np.round(np.clip(y / max_abs, -1.0, 1.0) * qmax).astype(np.int16)

        y_hat = q.astype(np.float32) / qmax * max_abs
        residual = y - y_hat
        residual_sign = (residual >= 0).astype(np.uint8)

        return {
            "q": q,
            "scale": max_abs,
            "residual_sign": residual_sign,
        }

    def decompress(self, packed: dict) -> np.ndarray:
        q = packed["q"]
        max_abs = packed["scale"]
        residual_sign = packed["residual_sign"]

        qmax = 2 ** self.bits // 2 - 1
        y_hat = q.astype(np.float32) / qmax * max_abs

        # crude toy residual correction
        correction = (residual_sign.astype(np.float32) * 2.0 - 1.0) * (max_abs / (8 * qmax + 1e-8))
        y_corr = y_hat + correction

        x_hat = y_corr @ self.R.T
        return x_hat

    def estimate_bits_per_value(self) -> float:
        # q uses self.bits bits, residual sign uses 1 bit, scale overhead amortized separately
        return float(self.bits + 1)


# -----------------------------
# KVTC-like toy codec
# -----------------------------
class KVTCToy:
    """
    Educational toy approximation:
    1) calibrate PCA on a sample
    2) decorrelate features
    3) allocate bits based on component variance
    4) scalar quantize each component

    This is NOT the real KVTC implementation.
    It captures the codec-style intuition only.
    """

    def __init__(self, dim: int, total_bits_per_vector: int):
        self.dim = dim
        self.total_bits_per_vector = total_bits_per_vector
        self.mean = None
        self.components = None
        self.component_bits = None
        self.scales = None

    def fit(self, calibration_data: np.ndarray) -> None:
        self.mean = calibration_data.mean(axis=0, keepdims=True)
        x = calibration_data - self.mean

        cov = np.cov(x, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        self.components = eigvecs

        # bit allocation proportional to variance
        weights = eigvals / (eigvals.sum() + 1e-8)
        raw = weights * self.total_bits_per_vector
        bits = np.floor(raw).astype(int)

        while bits.sum() < self.total_bits_per_vector:
            idx = int(np.argmax(raw - bits))
            bits[idx] += 1

        bits = np.maximum(bits, 1)
        self.component_bits = bits

        z = x @ self.components
        self.scales = np.max(np.abs(z), axis=0, keepdims=True) + 1e-8

    def compress(self, x: np.ndarray) -> dict:
        z = (x - self.mean) @ self.components
        codes = []
        for i in range(self.dim):
            b = int(self.component_bits[i])
            qmax = 2 ** (b - 1) - 1
            qmax = max(qmax, 1)
            zi = np.round(np.clip(z[:, i:i+1] / self.scales[:, i:i+1], -1.0, 1.0) * qmax).astype(np.int16)
            codes.append(zi)
        return {"codes": codes}

    def decompress(self, packed: dict) -> np.ndarray:
        zs = []
        for i in range(self.dim):
            b = int(self.component_bits[i])
            qmax = 2 ** (b - 1) - 1
            qmax = max(qmax, 1)
            zi = packed["codes"][i].astype(np.float32) / qmax * self.scales[:, i:i+1]
            zs.append(zi)
        z = np.concatenate(zs, axis=1)
        x_hat = z @ self.components.T + self.mean
        return x_hat

    def estimate_avg_bits_per_value(self) -> float:
        return float(self.total_bits_per_vector / self.dim)


def main() -> None:
    set_seed(0)

    n = 1024
    dim = 16

    # synthetic KV-like vectors with correlation
    base = np.random.randn(n, dim // 2)
    x = np.concatenate([base, base + 0.1 * np.random.randn(n, dim // 2)], axis=1).astype(np.float32)

    # TurboQuant-like toy
    tq = TurboQuantToy(dim=dim, bits=3)
    tq_packed = tq.compress(x)
    x_tq = tq.decompress(tq_packed)

    # KVTC-like toy
    calibration = x[:256]
    kvtc = KVTCToy(dim=dim, total_bits_per_vector=48)  # avg 3 bits/value
    kvtc.fit(calibration)
    kvtc_packed = kvtc.compress(x)
    x_kvtc = kvtc.decompress(kvtc_packed)

    print("=== Toy KV compression comparison ===")
    print(f"Input shape: {x.shape}")
    print()
    print("TurboQuant-like toy")
    print(f"  approx bits/value : {tq.estimate_bits_per_value():.2f}")
    print(f"  reconstruction MSE: {mse(x, x_tq):.6f}")
    print()
    print("KVTC-like toy")
    print(f"  avg bits/value    : {kvtc.estimate_avg_bits_per_value():.2f}")
    print(f"  reconstruction MSE: {mse(x, x_kvtc):.6f}")
    print()
    print("Interpretation:")
    print("- TurboQuant-like path is lightweight and easy to apply.")
    print("- KVTC-like path uses calibration + transform coding.")
    print("- They are solving related but not identical serving problems.")


if __name__ == "__main__":
    main()
