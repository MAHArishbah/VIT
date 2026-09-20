# Vision Transformer from Scratch on ImageNet-100

A Vision Transformer (ViT) implemented from scratch in PyTorch and trained from random initialization on ImageNet-100 (100 classes, ~117k training images), with controlled experiments tracked in Weights & Biases.

**Final model (test set, used once): 65.84% ± 0.67 top-1, 87.38% top-5.**
Regularization improved test top-1 by **+5.0 points** and cut the train–test generalization gap from **33.3 to 12.2 points**.

- Notebook: `ViT_Scratch.ipynb` · [Open in Colab](https://colab.research.google.com/drive/1loZnQcFo-yy-120RJet0cECpjTsB2unc#scrollTo=ez-5uZsV8EGp)
- Experiment tracking: [W&B project](https://wandb.ai/maharishabh7552-dsv-global-transport-and-logistics/ViT-imagenet100/overview)

---

## Highlights

- **Model written by hand:** convolutional patch embedding, CLS token, learned position embedding, pre-norm transformer blocks (attention via `nn.MultiheadAttention`), MLP head, full training loop.
- **Controlled experiments:** baseline → iteration 1 (optimization) → iteration 2 (regularization), fixed seed, one group of changes per run.
- **Honest evaluation:** model selection on validation only; untouched test split evaluated once at the end; standard errors on every reported accuracy.
- **Diagnostics beyond accuracy:** generalization gap measured directly on clean training images; bias–variance analysis; validation-vs-test discrepancy investigated.
- **Robust training on Colab:** per-run checkpoints on Google Drive, exact resume after disconnects (model, optimizer, scheduler, step counters, history), resumable W&B runs, model artifacts.

---

## Model

| Setting | Value |
|---|---|
| Input | 224 × 224 RGB |
| Patch size | 16 → 196 patches + 1 CLS = 197 tokens |
| Embedding dim (D) | 256 |
| Transformer blocks (L) | 6 |
| Attention heads | 8 (head dim 32) |
| MLP hidden size | 512 |
| Normalization | LayerNorm (pre-norm) |
| Classifier | LayerNorm + Linear on the CLS token |
| Parameters | 3,436,388 |
| Forward compute | ≈ 1.56 GFLOP per image (≈ 0.78 GMAC); training ≈ 4.7 GFLOP per image |

---

## Data

- Source: [`ilee0022/ImageNet100`](https://huggingface.co/datasets/ilee0022/ImageNet100) (Hugging Face version of the Kaggle ImageNet-100 subset of ILSVRC-2012).
- Splits: **train 117,000** (1,170/class) · **validation 13,000** (130/class) · **test 5,000** (50/class).
- Train + validation together are 1,300 images per class, i.e. ImageNet's *training* images for these classes split 90/10. The test split (50/class) is most likely ImageNet's official validation images, collected separately.
- Preprocessing: train = RandomResizedCrop(224, scale 0.08–1) + horizontal flip (+ RandAugment in iteration 2); validation/test = Resize(256) + CenterCrop(224). ImageNet mean/std normalization.

---

## Experiments

| | Baseline | Iteration 1 — optimization | Iteration 2 — regularization |
|---|---|---|---|
| Optimizer | Adam | AdamW, weight decay 0.05 (not on biases, LayerNorm, CLS, position embedding) | same as iteration 1 |
| Learning rate | constant 5e-4 | 5-epoch linear warmup → cosine decay to 1e-6 (per step), peak 5e-4 | same |
| Gradient clipping | — | global norm 1.0 | same |
| CLS / position init | `randn` (std 1) | truncated normal, std 0.02 | same |
| Epochs | 20 | 50 | 50 |
| Augmentation | crop + flip | crop + flip | + RandAugment (2 ops, magnitude 9) |
| Label smoothing | — | — | 0.1 |
| CutMix | — | — | α = 1.0 on 50% of batches |
| Dropout | — | — | code in place, set to 0.0 |
| Other | — | seed 42, `drop_last`, persistent workers, TF32 matmuls, fused attention (`need_weights=False`) | same |
| GPU | A100 | A100 | L4 |

Each iteration changes one *group* of settings, so iteration 1 → iteration 2 isolates the effect of regularization. Baseline → iteration 1 changes both the recipe and the training length, so that gain is "better recipe + longer training".

---

## Results

### Validation (used for all model comparisons and checkpoint selection)

| Run | Best val top-1 | Mean of last 3 epochs | Best val top-5 | Plateau* |
|---|---|---|---|---|
| Baseline (20 epochs) | 58.37692 | — | — | — |
| Iteration 1 | 66.53% (epoch 49) | 66.49% | 87.88% (epoch 43) | epoch 36 |
| Iteration 2 | **70.45%** (epoch 48) | 70.44% | **90.31%** (epoch 50) | epoch 42 |

\*First epoch with val accuracy within 1 point of the run's best.

### Test (evaluated once, after all decisions were made)

| Run | Test top-1 (± SE) | Test top-5 | Test loss |
|---|---|---|---|
| Iteration 1 | 60.82% ± 0.69 | 83.82% | 1.652 |
| **Iteration 2 (final)** | **65.84% ± 0.67** | **87.38%** | 1.362 |

Regularization gain on test: **+5.0 points top-1** (≈ 5 standard errors), +3.6 points top-5; top-1 error down 13% relative, top-5 error down 22% relative.

### Generalization gap, measured directly

Same checkpoints, same preprocessing, plain cross-entropy; 13,000 random *training* images vs held-out images.

| Run | Clean-train accuracy | Val accuracy | Gap vs val | Test accuracy | Gap vs test |
|---|---|---|---|---|---|
| Iteration 1 | 94.10% | 66.52% | 27.6 pts | 60.82% | 33.3 pts |
| Iteration 2 | 78.08% | 70.45% | 7.6 pts | 65.84% | 12.2 pts |

**Bias–variance reading.** Iteration 1 is low-bias, high-variance: it fits 94% of its training images but is far worse on new ones. Regularization raised training error (bias proxy, +16 pts) but cut the gap (variance proxy, −20 pts vs validation), so total error fell. Iteration 2 is now bias-dominated, and was still improving at the end of training (best epochs 48–50): the next gains should come from more training or a larger model rather than more regularization.

### Validation vs test

Both models score lower on test than on validation (iteration 1: −5.7 pts, iteration 2: −4.6 pts; each ≈ 6–7 standard errors). The validation split is a random slice of the same pool as the training images, while the test images were collected separately, so validation overstates absolute accuracy. The two drops differ by only 1.1 pts (≈ 1 standard error), so the ranking and the gap between the models are preserved: comparisons on validation remain valid, and absolute performance is reported on test.

### Statistics

- Single seed per configuration (compute budget); differences under ~2 points are treated as inconclusive.
- Standard errors from √(p(1−p)/n): ≈ 0.4 pts on validation (13,000 images), ≈ 0.67 pts on test (5,000 images). Standard errors of differences combine the individual ones and are conservative, since all models were evaluated on the same images.
- No hyperparameter search: optimizer settings follow the DeiT recipe (peak learning rate 5e-4, not rescaled for batch size 128).
- Iteration 1 (A100): the first ~10 epochs ran with 2 data-loading workers at ≈ 7.3 min/epoch; after switching to 8 workers (resume at epoch 11) ≈ 2.2 min/epoch — a ~3.4× speedup from the input pipeline alone. Total 2 h 49 min.
- Iteration 2 (L4, 8 workers throughout, plus RandAugment's extra CPU work): ≈ 2.7 min/epoch, total 2 h 15 min. The shorter total comes from the A100 run's slow 2-worker start; per epoch, the A100 with 8 workers was ~20% faster.
- Throughput: ≈ 1.3 TFLOPS with 2 workers vs ≈ 4.4 TFLOPS with 8 workers on the A100 — still only ~3% of its TF32 peak.

---

## Training details and compute

- Batch size 128, 914 steps per epoch, 45,700 steps for 50 epochs.
- Iteration 1 on an A100: 2 h 49 min for 50 epochs (≈ 3.4 min per epoch).
- Iteration 2 on an L4: `2h 14 min`.
- Estimated compute per epoch ≈ 5.7 × 10¹⁴ FLOP (≈ 2.8 × 10¹⁶ FLOP per 50-epoch run). Iteration 1 achieved ≈ 2.8 TFLOPS, about 2% of the A100's TF32 peak, which points to the input pipeline (JPEG decoding and augmentation with 2 workers) rather than GPU arithmetic as the bottleneck.
- Estimated activation memory ≈ 2 GB per batch of 128; checkpoints ≈ 41 MB (weights + AdamW state).

---

## Lessons learned

1. **Logged training metrics understate overfitting.** Accuracy on augmented training images (83%) hid a clean-train accuracy of 94% and a true gap of 27.6–33.3 points.
2. **Regularization pays off only when variance dominates.** It traded training fit for generalization; once the gap was small, further regularization would mostly add bias.
3. **A validation split carved from the training pool is optimistic.** It overstated accuracy by ~5 points versus separately collected test images, while preserving the ranking between runs.
4. **Cheap engineering changes matter.** TF32 and fused attention sped training up substantially with no change to the recipe; `drop_last` removed tiny 8-image final batches that caused loss/accuracy spikes.
5. **Resumability is part of the experiment.** Saving the optimizer, scheduler and counters (not just weights) made Colab disconnects cost at most one epoch.

---

## Reproducing

1. Open `ViT_Scratch.ipynb` in Google Colab with a GPU runtime (A100 or L4).
2. Add a Colab secret named `WANDB_API_KEY`.
3. Set `run_name` and `run_id` in the hyperparameter cell (a new pair per run), then **Runtime → Run all** and authorize Google Drive.
4. After a disconnect, rerun all cells unchanged: training resumes from the last completed epoch and the same W&B run continues.
5. **Find the bottleneck before paying for a bigger GPU.** With 2 data-loading workers the A100 spent most of its time waiting for data (≈ 7.3 min/epoch); 8 workers made it ≈ 3.4× faster with no change to the model or GPU.

All hyperparameters live in one cell; the seed (42) fixes initialization, data order and augmentation randomness.

---

## Limitations and next steps

- Single seed per configuration; no hyperparameter search.
- 50 epochs is short for a ViT trained from scratch (DeiT uses 300).
- Next: longer training or a larger model (the final model is bias-dominated); Mixup and stochastic depth; 2D rotary position embeddings (RoPE) as an architecture experiment; a paired per-image test (McNemar) for sharper comparisons.

---

## References

- Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021.
- Touvron et al., *Training data-efficient image transformers & distillation through attention* (DeiT), ICML 2021.
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW), ICLR 2019.
- Cubuk et al., *RandAugment*, CVPR Workshops 2020.
- Yun et al., *CutMix*, ICCV 2019.
- Szegedy et al., *Rethinking the Inception Architecture* (label smoothing), CVPR 2016.
- Russakovsky et al., *ImageNet Large Scale Visual Recognition Challenge*, IJCV 2015.
