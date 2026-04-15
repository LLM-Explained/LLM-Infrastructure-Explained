# Prefill–Decode Disaggregation Backbone

This repository provides a **core backbone** for the main systems idea behind **prefill–decode disaggregation**:

separate prompt processing and token generation into distinct worker pools, then connect them with explicit KV-cache transfer.

## What this code covers

This scaffold includes the major architectural pieces:

- a request model with:
  - prefill cost
  - decode length
  - KV transfer cost

- a **unified serving** path
  - one shared pool for prefill and decode

- a **disaggregated serving** path
  - separate prefill pool
  - separate decode pool
  - explicit KV transfer
  - simple overlap between phases

## Core architecture

The backbone implemented here is:

1. run prefill on a prefill-specialized pool
2. transfer KV cache to the decode pool
3. run decode on a decode-specialized pool
4. overlap work where possible

Conceptually:

```text
request -> prefill pool -> KV transfer -> decode pool
```
