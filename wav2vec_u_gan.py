"""
wav2vec_u_gan.py
================
Wav2Vec-U GAN implementation with separation of concerns.

Three clearly delineated classes:
  - RealData      : provides "real" samples (phoneme sequences from unpaired text)
  - Generator     : provides "fake" samples (phoneme predictions from audio features)
  - Discriminator : discriminates between "real" and "fake" samples

Reference: Baevski et al., "Unsupervised Speech Recognition", NeurIPS 2021.
           https://arxiv.org/abs/2105.11084
"""

import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


# ---------------------------------------------------------------------------
# RealData
# ---------------------------------------------------------------------------

class RealData:
    """
    Provides "real" samples for the discriminator.

    Real samples are phoneme sequences derived from an unpaired text corpus
    that has been phonemized (e.g. via G2P). The corpus is stored as a
    plain-text file with one space-separated phoneme sequence per line,
    produced by the prepare_text() stage of the pipeline.

    The discriminator never sees audio — it only sees:
      (a) one-hot phoneme vectors drawn from this class  (real)
      (b) soft phoneme probability vectors from Generator (fake)

    Args:
        phones_path : path to the phonemized text file
                      (e.g. data/text/phones/phones.txt)
        phoneme_vocab : ordered list of phoneme symbols (determines vocab size)
        batch_size    : number of sequences to sample per call to get_batch()
        max_len       : pad/truncate sequences to this length
        device        : torch device
    """

    def __init__(
        self,
        phones_path: str,
        phoneme_vocab: list,
        batch_size: int = 160,
        max_len: int = 512,
        device: str = "cpu",
    ):
        self.phoneme_vocab = phoneme_vocab
        self.vocab_size = len(phoneme_vocab)
        self.batch_size = batch_size
        self.max_len = max_len
        self.device = torch.device(device)

        # Map phoneme symbol → integer index
        self.sym2idx = {sym: i for i, sym in enumerate(phoneme_vocab)}

        # Load all phoneme sequences from file
        self.sequences = self._load(phones_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, path: str) -> list:
        """Read phonemized text file; return list of integer-index sequences."""
        sequences = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                phones = line.strip().split()
                if not phones:
                    continue
                idxs = [self.sym2idx[p] for p in phones if p in self.sym2idx]
                if idxs:
                    sequences.append(idxs)
        if not sequences:
            raise ValueError(f"No valid phoneme sequences found in {path}")
        return sequences

    def _to_one_hot(self, indices: list, length: int) -> torch.Tensor:
        """Convert a list of integer phone IDs to a one-hot tensor [T, V]."""
        seq = torch.tensor(indices[:length], dtype=torch.long)
        one_hot = F.one_hot(seq, num_classes=self.vocab_size).float()
        # Pad to max_len if shorter
        pad = length - one_hot.size(0)
        if pad > 0:
            one_hot = F.pad(one_hot, (0, 0, 0, pad))
        return one_hot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_batch(self) -> tuple:
        """
        Sample a batch of real phoneme sequences.

        Returns:
            x            : [B, max_len, vocab_size]  one-hot float tensor
            padding_mask : [B, max_len]  bool tensor, True where padded
        """
        sampled = random.choices(self.sequences, k=self.batch_size)
        lengths = [min(len(s), self.max_len) for s in sampled]

        one_hots = [self._to_one_hot(s, self.max_len) for s in sampled]
        x = torch.stack(one_hots, dim=0).to(self.device)  # [B, T, V]

        # Padding mask: True at positions that are padding
        padding_mask = torch.zeros(self.batch_size, self.max_len, dtype=torch.bool)
        for i, l in enumerate(lengths):
            padding_mask[i, l:] = True
        padding_mask = padding_mask.to(self.device)

        return x, padding_mask

    def vocab_size_(self) -> int:
        return self.vocab_size


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """
    Provides "fake" samples for the discriminator.

    Maps segment-level audio feature vectors (PCA-reduced wav2vec 2.0
    representations) to a distribution over phonemes at each time step.

    Architecture: optional BatchNorm → Dropout → 1D Conv → (optionally) Gumbel-Softmax

    The Generator is intentionally lightweight (single Conv1d layer). The
    heavy representation learning is done upstream by wav2vec 2.0; the
    Generator only needs to learn a shallow mapping from the already-rich
    acoustic representations to the phoneme vocabulary.

    Args:
        input_dim    : dimension of input segment features (default: 512 from PCA)
        output_dim   : number of phonemes in vocabulary (e.g. 41 for English)
        kernel_size  : Conv1d kernel size (default: 4, from original paper)
        dropout      : dropout probability (default: 0.1)
        use_batchnorm: whether to apply BatchNorm before Conv (default: True)
    """

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 41,
        kernel_size: int = 4,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.batch_norm = nn.BatchNorm1d(input_dim) if use_batchnorm else None
        self.dropout = nn.Dropout(p=dropout)

        # Causal padding so output length == input length
        # For a kernel of size k: left-pad by (k-1) on the time axis
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=0,   # We apply manual causal padding below
            bias=False,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _causal_pad(self, x: torch.Tensor) -> torch.Tensor:
        """Left-pad the time dimension so Conv1d output length == input length."""
        return F.pad(x, (self.pad, 0))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            features     : [B, T, input_dim]  — PCA-reduced segment embeddings
            padding_mask : [B, T]  bool tensor, True where padded (optional)

        Returns dict with:
            dense_x  : [B, T, output_dim]  softmax probabilities (differentiable)
            token_x  : [B, T, output_dim]  one-hot hard assignments (argmax)
            logits   : [B, T, output_dim]  raw logits (used for smoothness loss)
        """
        B, T, C = features.shape

        x = features  # [B, T, C]

        # BatchNorm expects [B, C, T]
        if self.batch_norm is not None:
            x = x.transpose(1, 2)          # [B, C, T]
            x = self.batch_norm(x)
            x = x.transpose(1, 2)          # [B, T, C]

        x = self.dropout(x)

        # Conv1d expects [B, C, T]
        x = x.transpose(1, 2)              # [B, C, T]
        x = self._causal_pad(x)            # [B, C, T + pad]
        x = self.conv(x)                   # [B, output_dim, T]
        x = x.transpose(1, 2)             # [B, T, output_dim]

        logits = x  # raw logits [B, T, output_dim]

        # Compute softmax on raw logits; then zero out padded positions.
        # (Applying -inf before softmax causes NaN when all positions in a row
        # are masked, because softmax(-inf,...,-inf) = 0/0.)
        dense_x = torch.softmax(logits, dim=-1)  # [B, T, output_dim]
        if padding_mask is not None:
            dense_x = dense_x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # Hard one-hot assignments via argmax (straight-through in backward pass)
        indices = logits.argmax(dim=-1)           # [B, T]
        token_x = F.one_hot(indices, num_classes=self.output_dim).float()

        return {
            "dense_x": dense_x,   # differentiable probabilities
            "token_x": token_x,   # discrete one-hot
            "logits": logits,      # raw logits (for smoothness loss)
        }


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    """
    Discriminates between "real" phoneme sequences and "fake" ones.

    Takes a sequence of phoneme probability vectors (one-hot for real data,
    softmax outputs for generated data) and predicts a scalar real/fake score.

    Architecture: Linear projection → [causal Conv1d → GELU → Dropout] × n_layers
                  → mean pool over time → scalar output

    The discriminator is causal (each position attends only to past context)
    to model the sequential/temporal structure of phoneme sequences.

    Args:
        input_dim  : phoneme vocabulary size (same as Generator output_dim)
        hidden_dim : inner Conv1d channel dimension (default: 384)
        kernel_size: Conv1d kernel size (default: 6)
        n_layers   : number of inner Conv1d blocks (default: 2)
        dropout    : dropout probability (default: 0.1)
        causal     : if True, use causal (left-only) padding (default: True)
    """

    def __init__(
        self,
        input_dim: int = 41,
        hidden_dim: int = 384,
        kernel_size: int = 6,
        n_layers: int = 2,
        dropout: float = 0.1,
        causal: bool = True,
    ):
        super().__init__()
        self.causal = causal
        self.kernel_size = kernel_size

        # Initial projection: phoneme vocab → hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Stack of causal Conv1d blocks
        self.conv_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.conv_layers.append(
                nn.Sequential(
                    nn.GELU(),
                    nn.Dropout(p=dropout),
                    nn.Conv1d(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        kernel_size=kernel_size,
                        padding=0,  # manual causal padding applied in forward
                        bias=False,
                    ),
                )
            )

        # Final linear: hidden_dim → 1 scalar
        self.output_proj = nn.Linear(hidden_dim, 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _causal_pad(self, x: torch.Tensor) -> torch.Tensor:
        """Left-pad the time dimension for causal convolution."""
        if self.causal:
            return F.pad(x, (self.kernel_size - 1, 0))
        else:
            # Symmetric padding to preserve length
            p = (self.kernel_size - 1) // 2
            return F.pad(x, (p, self.kernel_size - 1 - p))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x            : [B, T, vocab_size]  phoneme probability vectors
            padding_mask : [B, T]  bool tensor, True where padded (optional)

        Returns:
            scores : [B]  scalar real/fake logit per sequence
        """
        # Project to hidden dim: [B, T, hidden_dim]
        x = self.input_proj(x)

        # Transpose for Conv1d: [B, hidden_dim, T]
        x = x.transpose(1, 2)

        for conv_block in self.conv_layers:
            residual = x
            x = self._causal_pad(x)   # [B, hidden_dim, T + pad]
            x = conv_block(x)          # [B, hidden_dim, T]
            x = x + residual           # residual connection

        # Back to [B, T, hidden_dim]
        x = x.transpose(1, 2)

        # Mask padded positions before pooling
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            # Mean over non-padded positions
            lengths = (~padding_mask).sum(dim=1, keepdim=True).float().clamp(min=1)
            x = x.sum(dim=1) / lengths   # [B, hidden_dim]
        else:
            x = x.mean(dim=1)            # [B, hidden_dim]

        scores = self.output_proj(x).squeeze(-1)  # [B]
        return scores


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def adversarial_loss(
    discriminator: Discriminator,
    real_x: torch.Tensor,
    fake_x: torch.Tensor,
    real_mask: Optional[torch.Tensor] = None,
    fake_mask: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> tuple:
    """
    Binary cross-entropy adversarial loss for both discriminator and generator.

    Discriminator maximises: E[D(real)] - E[D(fake)]
    Generator minimises:    -E[D(fake)]   (i.e. tries to fool discriminator)

    Returns:
        d_loss : discriminator loss (real + fake combined)
        g_loss : generator loss
    """
    real_scores = discriminator(real_x, real_mask)   # [B]
    fake_scores = discriminator(fake_x, fake_mask)   # [B]

    real_labels = torch.ones_like(real_scores) - label_smoothing
    fake_labels = torch.zeros_like(fake_scores) + label_smoothing

    d_loss = (
        F.binary_cross_entropy_with_logits(real_scores, real_labels)
        + F.binary_cross_entropy_with_logits(fake_scores, fake_labels)
    )

    # Generator tries to make discriminator output "real" for fake samples
    g_loss = F.binary_cross_entropy_with_logits(
        fake_scores, torch.ones_like(fake_scores)
    )

    return d_loss, g_loss


def gradient_penalty(
    discriminator: Discriminator,
    real_x: torch.Tensor,
    fake_x: torch.Tensor,
    real_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    WGAN-GP gradient penalty on interpolations between real and fake samples.

    Encourages the discriminator to be 1-Lipschitz, which stabilises training
    and prevents mode collapse.

        L_gp = E[(‖∇_x̂ D(x̂)‖₂ − 1)²]    where x̂ = α·real + (1−α)·fake

    Returns:
        penalty : scalar tensor
    """
    B = real_x.size(0)
    # Random interpolation coefficient per sample
    alpha = torch.rand(B, 1, 1, device=real_x.device)

    # Interpolate; ensure same length (truncate to shorter)
    T = min(real_x.size(1), fake_x.size(1))
    interpolated = alpha * real_x[:, :T] + (1 - alpha) * fake_x[:, :T]
    interpolated = interpolated.requires_grad_(True)

    mask = real_mask[:, :T] if real_mask is not None else None
    scores = discriminator(interpolated, mask)   # [B]

    gradients = autograd.grad(
        outputs=scores.sum(),
        inputs=interpolated,
        create_graph=True,
        retain_graph=True,
    )[0]  # [B, T, V]

    # Norm over the (T, V) dimensions, then penalise deviation from 1
    grad_norm = gradients.reshape(B, -1).norm(2, dim=1)  # [B]
    penalty = ((grad_norm - 1) ** 2).mean()
    return penalty


def smoothness_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Penalises temporally inconsistent phoneme predictions from the Generator.

    Consecutive logit frames should not change drastically, encouraging the
    model to produce smooth, speech-like phoneme sequences.

        L_sp = (1/(T−1)) · Σ_t ‖logits_t − logits_{t+1}‖²

    Args:
        logits : [B, T, V]  raw generator logits

    Returns:
        loss : scalar tensor
    """
    return F.mse_loss(logits[:, :-1], logits[:, 1:])


def code_penalty(dense_x: torch.Tensor) -> torch.Tensor:
    """
    Penalises the Generator for collapsing to a small subset of phonemes
    (mode collapse in the discrete phoneme space).

    Computed as the normalised shortfall from maximum perplexity:

        L_pd = (|V| − exp(H(p̄))) / |V|

    where p̄ is the batch-averaged phoneme distribution and H(·) is entropy.
    When all phonemes are used equally, perplexity == |V| and loss == 0.

    Args:
        dense_x : [B, T, V]  generator softmax probabilities

    Returns:
        penalty : scalar tensor
    """
    V = dense_x.size(-1)
    avg_probs = dense_x.mean(dim=[0, 1])  # [V]  batch-and-time average
    entropy = -(avg_probs * (avg_probs + 1e-8).log()).sum()
    perplexity = entropy.exp()
    return (V - perplexity) / V


# ---------------------------------------------------------------------------
# Standalone training demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import numpy as np
    from tensorboardX import SummaryWriter

    TB_DIR = "runs/standalone_gan"
    writer = SummaryWriter(TB_DIR)
    print(f"TensorBoard logs → {TB_DIR}")

    DATA_ROOT = os.path.expanduser(
        "~/wav2vec_unsupervised/data/clustering/librispeech/"
        "precompute_pca512_cls128_mean_pooled"
    )
    TEXT_ROOT = os.path.expanduser(
        "~/wav2vec_unsupervised/data/text/phones"
    )
    BATCH_SIZE = 16
    MAX_LEN = 256
    N_STEPS = 200
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    # 1. Build vocab from dict.phn.txt
    vocab = [line.split()[0] for line in open(f"{TEXT_ROOT}/dict.phn.txt")]
    vocab_size = len(vocab)
    print(f"Phoneme vocab size: {vocab_size}")

    # 2. Load RealData from lm.phones.filtered.txt
    real_data = RealData(
        phones_path=f"{TEXT_ROOT}/lm.phones.filtered.txt",
        phoneme_vocab=vocab,
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
        device=DEVICE,
    )
    print(f"Real phoneme sequences loaded: {len(real_data.sequences)}")

    # 3. Load audio features (flat array → list of per-sequence arrays)
    features_flat = np.load(f"{DATA_ROOT}/train.npy")          # (N_frames, 512)
    lengths = [int(x) for x in open(f"{DATA_ROOT}/train.lengths")]
    sequences = []
    offset = 0
    for l in lengths:
        sequences.append(features_flat[offset : offset + l])
        offset += l
    print(f"Audio sequences loaded: {len(sequences)}, total frames: {features_flat.shape[0]}")

    # 4. Instantiate model components
    gen  = Generator(input_dim=512, output_dim=vocab_size).to(DEVICE)
    disc = Discriminator(input_dim=vocab_size).to(DEVICE)
    opt_g = torch.optim.Adam(gen.parameters(),  lr=4e-5, betas=(0.5, 0.98))
    opt_d = torch.optim.Adam(disc.parameters(), lr=2e-5, betas=(0.5, 0.98))
    print(f"Generator params:     {sum(p.numel() for p in gen.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in disc.parameters()):,}")
    print()

    # 5. Training loop — alternate D and G updates each step
    for step in range(1, N_STEPS + 1):
        # Sample a minibatch of audio sequences
        idxs = random.sample(range(len(sequences)), BATCH_SIZE)
        batch_seqs = [sequences[i] for i in idxs]

        # Pad to MAX_LEN
        audio    = np.zeros((BATCH_SIZE, MAX_LEN, 512), dtype=np.float32)
        pad_mask = np.ones((BATCH_SIZE, MAX_LEN), dtype=bool)
        for i, s in enumerate(batch_seqs):
            l = min(len(s), MAX_LEN)
            audio[i, :l]    = s[:l]
            pad_mask[i, :l] = False

        audio_t = torch.tensor(audio).to(DEVICE)
        pad_t   = torch.tensor(pad_mask).to(DEVICE)

        gen_out          = gen(audio_t, pad_t)
        real_x, real_msk = real_data.get_batch()

        if step % 2 == 1:   # ---------- Discriminator step ----------
            opt_d.zero_grad()
            d_loss, _ = adversarial_loss(disc, real_x, gen_out["dense_x"].detach(),
                                         real_msk, pad_t)
            gp   = gradient_penalty(disc, real_x, gen_out["dense_x"].detach(), real_msk)
            loss = d_loss + 0.5 * gp
            loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 5.0)
            opt_d.step()
            writer.add_scalar("discriminator/d_loss", d_loss.item(), step)
            writer.add_scalar("discriminator/gradient_penalty", gp.item(), step)
            writer.add_scalar("discriminator/total_loss", loss.item(), step)
            if step % 10 == 1:
                print(f"Step {step:4d} | D_loss={d_loss.item():.4f}  GP={gp.item():.4f}")

        else:               # ---------- Generator step ----------
            opt_g.zero_grad()
            _, g_loss = adversarial_loss(disc, real_x, gen_out["dense_x"],
                                          real_msk, pad_t)
            sp   = smoothness_loss(gen_out["logits"])
            cp   = code_penalty(gen_out["dense_x"])
            loss = g_loss + 1.5 * sp + 4.0 * cp
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 5.0)
            opt_g.step()
            writer.add_scalar("generator/g_loss", g_loss.item(), step)
            writer.add_scalar("generator/smoothness", sp.item(), step)
            writer.add_scalar("generator/code_penalty", cp.item(), step)
            writer.add_scalar("generator/total_loss", loss.item(), step)
            if step % 10 == 0:
                print(f"Step {step:4d} | G_loss={g_loss.item():.4f}  "
                      f"Smooth={sp.item():.4f}  CodePen={cp.item():.4f}")

    writer.close()
    print()
    print("Done. wav2vec_u_gan.py ran independently.")
    print(f"TensorBoard: tensorboard --logdir {TB_DIR} --port 6007")

