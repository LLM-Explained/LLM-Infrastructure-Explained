# Toy TurboQuant vs KVTC Demo

This is a minimal runnable demo illustrating two different KV-cache compression philosophies:

- a **TurboQuant-like** lightweight, training-free path
- a **KVTC-like** calibration-based transform-coding path

## What this demo is

An educational prototype that compares two toy codecs on synthetic KV-like vectors.

It is intended to illustrate the systems intuition:

- lightweight low-overhead KV compression
- versus heavier transform-coding-oriented KV storage compression

## What this demo is NOT

This is **not** an implementation of Google's TurboQuant or NVIDIA's KVTC.

It does **not** reproduce:
- PolarQuant
- QJL
- the real TurboQuant estimator
- the real KVTC PCA / adaptive quantization / entropy coding pipeline
- any production inference kernel
- any benchmark-quality measurements

It only captures the broad design ideas in a tiny runnable form.

## Requirements

- Python 3.9+
- NumPy

## Install

```bash
pip install numpy
```
