# KV Lifecycle-Aware Codec Demo

This is a minimal runnable demo showing the core idea that KV compression should depend on KV lifecycle.

It simulates three KV cache states:

- hot
- warm
- cold

and assigns a different codec policy to each one.

## What this demo shows

- active KV blocks become **hot**
- inactive but high-reuse blocks become **warm**
- inactive and low-reuse blocks become **cold**

Then each tier gets a different compression strategy:

- hot → low-overhead codec
- warm → balanced codec
- cold → higher-compression codec

## What this demo is NOT

This is not an implementation of TurboQuant, KVTC, LMCache, Mooncake, or Dynamo.

It does not implement:

- real KV tensors
- real quantization
- entropy coding
- PCA
- GPU memory movement
- serving-engine integration

It only demonstrates the policy idea in the smallest runnable form.

## Requirements

- Python 3.9+

No extra dependencies are required.

## Run

```bash
python demo.py
```
