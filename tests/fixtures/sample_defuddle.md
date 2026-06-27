# Attention and Context: A Unified Framework for Sequential Prediction

## Abstract

The dominant approach to sequence transduction tasks has long relied on recurrent
architectures that process tokens one step at a time. This sequential coupling
introduces a fundamental bottleneck: information from early positions in the
sequence must travel through a chain of hidden states before it can influence
predictions at later positions. We propose a fully attention-based encoder-decoder
that replaces recurrence entirely with multi-head self-attention layers, allowing
all positions to interact in $O(1)$ sequential operations. On the WMT 2014
English-to-German translation benchmark, our model achieves 28.4 BLEU, surpassing
the previous best result — including ensembles — by more than 2 BLEU points. The
same architecture reaches 41.0 BLEU on English-to-French with only a fraction of
the training cost. We further demonstrate strong transfer to syntactic parsing,
suggesting that attention-based representations generalise well beyond the tasks
for which they were originally designed.[^1]

## 1 Introduction

Recurrent neural networks — in particular long short-term memory (LSTM) units and
gated recurrent units (GRU) — have established strong baselines across a wide
range of sequence modelling tasks, including language modelling, machine translation,
and speech recognition. The hidden state $h_t$ of an RNN at time step $t$ is
computed from the previous hidden state $h_{t-1}$ and the current input; this
sequential dependency prevents parallel computation during training and during
inference whenever the input sequence is long.

Attention mechanisms were introduced to allow the decoder to look directly at any
part of the encoder output, bypassing the need to compress the entire source
sequence into a single fixed-length vector. Work by Bahdanau et al.[^2] showed that
soft alignment of target positions to source positions yields substantial
improvements on long sentences. Subsequent work extended these ideas to
self-attention within a single sequence, allowing each position to attend to all
others simultaneously.

### 1.1 Limitations of Sequential Models

A core difficulty with sequential models is that the number of operations required
to relate signals from two arbitrary positions grows with their distance in the
sequence. In convolutional sequence-to-sequence models this distance is $O(\log_k n)$
where $k$ is the kernel width and $n$ is the sequence length; in recurrent models
it is $O(n)$. Attention-based models reduce this to $O(1)$, at the cost of
quadratic memory in the sequence length. For most practical sequence lengths
encountered in natural language processing this trade-off is highly favourable.

The present work builds directly on this observation. Rather than using attention
as a supplement to recurrence, we use it as the sole mechanism for relating
representations of different positions, in both the encoder and the decoder.

## 2 Model Architecture

The model consists of stacked self-attention and point-wise fully connected layers
for both the encoder and the decoder. The encoder maps an input sequence of symbol
representations to a sequence of continuous representations. Given this sequence,
the decoder generates an output sequence one element at a time.

Each encoder layer has two sub-layers. The first is a multi-head self-attention
mechanism; the second is a simple position-wise fully connected feed-forward
network. We employ a residual connection around each sub-layer, followed by layer
normalisation, so that the output of each sub-layer is $\text{LayerNorm}(x + \text{Sub}(x))$.

The decoder inserts a third sub-layer that performs multi-head attention over the
output of the encoder stack. The self-attention sub-layer in the decoder is also
modified so that positions can attend only to earlier positions in the output
sequence. This masking ensures that the predictions for position $i$ depend only
on the known outputs at positions less than $i$.

### 2.1 Multi-Head Attention

Instead of performing a single attention function with $d_{\text{model}}$-dimensional
keys, values, and queries, we project the queries, keys, and values $h$ times with
different learned linear projections to $d_k$, $d_k$, and $d_v$ dimensions,
respectively. Attention is then applied in parallel to each of the $h$ projected
representations. The outputs are concatenated and once more projected, resulting in
the final value.

Multi-head attention allows the model to jointly attend to information from
different representation subspaces at different positions. With a single attention
head, averaging over these subspaces would suppress the distinct signals that each
head learns to capture.[^3]
