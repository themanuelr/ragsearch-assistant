---
title: "Sparse Attention Mechanisms for Long-Document Retrieval"
source: "https://arxiv.org/html/2406.01234"
language: "en"
word_count: 412
---
Provided proper attribution is provided, the authors hereby grant permission to use and
distribute this work in any form for non-commercial research purposes.

Jane Smith, Robert Lee, Ana Perez — Institute for Computational Research, 2024.

## Abstract

Modern transformer models apply attention across all token pairs, incurring
$O(n^2)$ memory and compute costs for sequence length $n$. This overhead becomes
prohibitive for long scientific documents where $n$ exceeds tens of thousands of
tokens. We propose Sparse Attention Mechanisms (SAM), a family of structured
sparse patterns that reduce attention complexity to $O(n \log n)$ without
sacrificing retrieval accuracy on benchmark datasets.[^1]

## 1 Introduction

Retrieval from long documents poses a central challenge for dense encoder models.
Standard self-attention operates over all pairwise interactions, which scales
quadratically with document length. For scientific literature — often exceeding
8 000 tokens — this constraint forces practitioners to truncate or chunk inputs,
potentially discarding sections critical to answering complex queries.

Prior work has explored local windows, global tokens, and learned sparse patterns
to mitigate this cost. We unify these approaches under a single framework and
evaluate them on three long-document retrieval benchmarks.

### 1.1 Contributions

We make three contributions. First, we introduce a plug-in sparse attention kernel
compatible with any HuggingFace encoder. Second, we release a long-document
retrieval benchmark with 10 000 annotated pairs drawn from scientific literature.
Third, we demonstrate that $O(n \log n)$ sparse attention matches dense baselines
on our benchmark while reducing memory consumption by 42 % at length 16 384.
