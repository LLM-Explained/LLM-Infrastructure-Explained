# MegaTrain Memory-Centric Training Backbone

This repository provides a **core backbone** for the main systems idea behind **MegaTrain**:

large-model training can be organized as a **memory-centric streaming system**, where persistent model state lives in host memory and the GPU acts as a transient compute engine.

## What this code covers

This scaffold includes the major architectural and algorithmic pieces:

- **host-resident parameters**
- **host-resident optimizer state**
- **stateless layer templates**
- **streamed dynamic weight binding**
- **simplified double-buffered prefetch structure**
- **host-side gradient update of master weights**

## Core architecture

The backbone implemented here is:

1. keep parameters and optimizer state in host memory
2. stream one layer’s parameters to device when needed
3. execute the stateless layer computation with dynamically bound weights
4. prefetch the next layer while computing the current one
5. return gradients to host
6. update the host-resident master weights

Conceptually:

```text
host params / optimizer state
        ↓ stream
GPU transient compute
        ↓ grads
host master update
```
