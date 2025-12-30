#!/bin/bash

echo "🚀 Running AdaFLUX-LoRA Simulation..."

python3 src/fedseq_adaflux_lora.py \
    --rounds 10 \
    --fit_clients 3 \
    --eval_clients 6

echo "📊 Launch TensorBoard with:"
echo "tensorboard --logdir runs/"
