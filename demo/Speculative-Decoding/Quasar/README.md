# Quasar--Inspired Verification Quantization Demo

This is a minimal runnable demo illustrating the core idea behind Quasar:

- speculative decoding can shift the bottleneck to the **verification stage**
- verification is memory-bandwidth heavy
- applying low-bit quantization to **verification only** can reduce memory traffic while preserving verification behavior reasonably well

## What this demo shows

The demo compares:

- a full-precision verification pass
- an 8-bit quantized verification pass
- a 4-bit quantized verification pass

It reports:

- logit MSE versus full precision
- agreement of verification decisions with full precision
- a simple memory-traffic proxy based on weight storage

## What this demo is

A tiny educational prototype for the *idea* of verification-stage quantization.

## What this demo is NOT

This is not Quasar.

It does **not** implement:

- real self-speculative decoding
- real verification kernels
- real hardware throughput measurements
- the full Quasar pipeline
- production-quality low-bit quantization
- acceptance-length evaluation on real LLMs

It only captures the central systems intuition in the smallest runnable form.

## Requirements

- Python 3.9+
- NumPy

## Install

```bash
pip install numpy
```
