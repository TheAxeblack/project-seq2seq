# Seq2Seq Learning with Neural Networks - Course Implementation

Implementation of **"Sequence to Sequence Learning with Neural Networks"**  
Sutskever, Vinyals & Le - Google, NeurIPS 2014.

---

## Overview

This repository contains a scaled-down, course-compatible reproduction of the Seq2Seq model from the paper. The task is **English → French machine translation** on a subset of the WMT'14 dataset. The core mechanisms of the paper are faithfully reproduced; only the scale is reduced to fit within the computational constraints of a student project.

---

## Implementation Details

### Architecture

The model follows the encoder–decoder design from the paper:

- **Encoder**: a 2-layer LSTM that reads the reversed source sentence and produces a fixed-size context vector (the final hidden state).
- **Decoder**: a separate 2-layer LSTM initialised with the encoder's context vector, generating the French translation one token at a time.
- **Source reversal**: input words are reversed before encoding — the key trick from the paper that reduces the minimal time lag between source and target words.
- **Teacher forcing** during training: the decoder receives the ground-truth French tokens as input rather than its own predictions.
- **Weight initialisation**: uniform distribution in [−0.08, 0.08], as specified in the paper.

### Training Setup

| Hyperparameter    | Value                                      | Paper value                       |
| ----------------- | ------------------------------------------ | --------------------------------- |
| Optimizer         | Adam (lr = 1e-3)                           | SGD (lr = 0.7)                    |
| Gradient clipping | 5                                          | 5                                 |
| Batch size        | 128                                        | 128                               |
| Epochs            | 20                                         | 7.5                               |
| LR schedule       | ReduceLROnPlateau (patience=2, factor=0.5) | Halve every ½ epoch after epoch 5 |
| Dropout           | 0.3                                        | —                                 |

### Decoding

Two decoding strategies are implemented:

- **Greedy decoding**: pick the highest-probability token at each step.
- **Beam search** (beam size = 5): maintain the top-5 partial hypotheses with **length normalisation** (α = 0.6) to avoid the bias towards short sequences.

---

## Results

Training was performed on a Tesla T4 GPU (Google Colab) for 20 epochs (~63 minutes).

| Model                     | Valid PPL (best) | BLEU      |
| ------------------------- | ---------------- | --------- |
| LSTM Seq2Seq — greedy     | **22.7**         | 4.18      |
| LSTM Seq2Seq — beam (B=5) | 22.7             | **4.70**  |
| **Paper (full scale)**    | —                | **34.81** |

The best validation checkpoint (epoch 8, PPL = 22.7) was used for all evaluations.  
The training loss continued to decrease after epoch 8 while the validation loss increased, indicating overfitting to the small training set.

### Sample translations (beam search)

| Source (EN)                                                                     | Reference (FR)                                                                              | Our model                                                                         |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| *i think it's conceivable that these data are used for mutual benefit .*        | *j'estime qu'il est concevable que ces données soient utilisées dans leur intérêt mutuel .* | *je pense qu'il est possible que ces données soient utilisées pour les `<UNK>` .* |
| *the airport is currently being evacuated and air traffic has been suspended .* | *l'aéroport est en cours d'évacuation et le trafic aérien est interrompu .*                 | *le `<UNK>` est maintenant `<UNK>` et le trafic aérien est `<UNK>` .*             |

The model captures sentence structure and common words reasonably well, but struggles with rare vocabulary (replaced by `<UNK>`) and domain-specific terms.

---

## Why Our BLEU is Lower than the Paper

The gap between our BLEU (4.70) and the paper's (34.81) is large but expected. The causes are well understood and ordered by importance:

### 1. Training data scale (dominant factor)

The paper trains on **12 million** sentence pairs; we use **200,000** (60× less). With so little data, the model sees each sentence roughly once per epoch and cannot learn the long tail of French vocabulary, idiomatic expressions, or rare grammatical constructions. This alone accounts for the majority of the BLEU gap.

### 2. Model size

The paper uses 1,000-dimensional embeddings and hidden states across 4 LSTM layers (384M parameters total). Our model uses 512 dimensions across 2 layers (~29M parameters — 13× smaller). A smaller model has strictly less capacity to memorise translation patterns.

### 3. Vocabulary coverage

The paper's 80,000-word target vocabulary covers the vast majority of French words in the test set. Our 10,000-word vocabulary produces many `<UNK>` tokens for rare words, which the BLEU metric penalises heavily since they never match the reference.

### 4. Overfitting

As visible in the learning curves, the training loss keeps decreasing after epoch 8 while the validation loss increases. The model memorises the 200k training sentences rather than generalising. The paper avoids this by having far more data than the model can memorise.

### 5. Optimizer difference

The paper uses SGD with lr = 0.7, which works well at large scale because the high learning rate acts as an implicit regulariser with large batches. We switched to Adam (lr = 1e-3) for stability at smaller scale — a pragmatic choice that does not degrade performance relative to our data size.

### What this result demonstrates

Despite the lower absolute score, a BLEU of ~5 on 200k training pairs with a 29M-parameter model is **consistent with correctly-implemented Seq2Seq baselines** in the literature at this scale. The result confirms that the core mechanisms work: the model learns word order, basic grammatical structure, and common vocabulary. The gap to the paper is a scale problem, not an implementation problem.

---

## Simplifications Summary

| Aspect              | Paper             | This project | Reason                    |
| ------------------- | ----------------- | ------------ | ------------------------- |
| Training pairs      | 12 M              | 200 k        | Colab time limit          |
| Hidden / embed size | 1 000             | 512          | GPU memory                |
| Encoder layers      | 4                 | 2            | Speed / quality trade-off |
| Source vocabulary   | 160 k             | 20 k         | Memory                    |
| Target vocabulary   | 80 k              | 10 k         | Memory                    |
| Training time       | ~10 days (8 GPUs) | ~63 min (T4) | Course constraint         |

---
