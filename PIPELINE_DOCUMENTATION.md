# Wav2Vec-U Pipeline — Full Technical Documentation

## Overview

This pipeline implements **Wav2Vec-U** (Baevski et al., NeurIPS 2021), an unsupervised speech recognition system. It trains a GAN to produce phoneme transcriptions from raw audio without ever seeing a single labelled (audio, transcript) pair. Instead it uses:

- **Unlabelled audio** — raw speech recordings
- **Unpaired text** — any text in the target language (no alignment to audio needed)

The pipeline takes raw `.wav` files and a plain-text corpus as inputs and produces phoneme-level transcriptions as output. It is split across several bash scripts that each handle a specific concern, orchestrated in sequence.

---

## Architecture: Files and Their Roles

```
run_pipeline.sh          ← entry point (hardcoded paths, calls run_wav2vec.sh)
run_wav2vec.sh           ← orchestrator for data prep stages
run_gans.sh              ← orchestrator for GAN training
run_eval.sh              ← orchestrator for evaluation/decoding

wav2vec_functions.sh     ← all data prep step implementations
gans_functions.sh        ← GAN training step implementation
eval_functions.sh        ← Viterbi decoding step implementation
utils.sh                 ← shared config, paths, checkpoint system, helpers

setup_functions.sh       ← one-time dependency installation
run_setup.sh             ← orchestrator for setup

wav2vec_u_gan.py         ← GAN model classes (Generator, Discriminator, RealData)
vads.py                  ← Python wrapper for rVADfast voice activity detection
```

---

## `utils.sh` — Shared Configuration and Checkpoint System

Every script in the pipeline sources `utils.sh` first. It serves two purposes:

### 1. Path Configuration

All directory and file paths used across the entire pipeline are defined here as bash variables. Key paths:

| Variable | Path | Purpose |
|---|---|---|
| `DIR_PATH` | `$HOME/wav2vec_unsupervised` | Root install directory |
| `DATA_ROOT` | `$DIR_PATH/data` | All generated pipeline outputs |
| `FAIRSEQ_ROOT` | `$DIR_PATH/fairseq_` | Fairseq toolkit (Ashesi custom fork) |
| `KENLM_ROOT` | `$DIR_PATH/kenlm/build/bin` | KenLM language model toolkit |
| `RVAD_ROOT` | `$DIR_PATH/rVADfast/src/rVADfast` | rVADfast silence detector |
| `MANIFEST_DIR` | `$DATA_ROOT/manifests` | Raw audio manifest TSV files |
| `NONSIL_AUDIO` | `$DATA_ROOT/processed_audio/` | Audio files with silence stripped |
| `MANIFEST_NONSIL_DIR` | `$DATA_ROOT/manifests_nonsil` | Manifests pointing to silence-free audio |
| `CLUSTERING_DIR` | `$DATA_ROOT/clustering/librispeech` | Audio features, cluster IDs, PCA outputs |
| `TEXT_OUTPUT` | `$DATA_ROOT/text` | Phonemized text, KenLM language model |
| `RESULTS_DIR` | `$DATA_ROOT/results/librispeech` | GAN training logs |
| `GANS_OUTPUT_PHONES` | `$DATA_ROOT/transcription_phones` | Final phoneme transcriptions |
| `MODEL` | `$DIR_PATH/pre-trained/wav2vec_vox_new.pt` | Pretrained wav2vec 2.0 checkpoint |
| `FASTTEXT_LIB_MODEL` | `$DIR_PATH/lid_model/lid.176.bin` | FastText language ID model |

Tunable parameters also live here:
- `NEW_SAMPLE_PCT=0.5` — fraction of audio used to fit k-means clusters
- `NEW_BATCH_SIZE=32` — batch size for wav2vec feature extraction
- `PHONEMIZER="G2P"` — which grapheme-to-phoneme tool to use
- `LANG="en"` — language code for text preparation
- `MIN_PHONES=3` — minimum phoneme sequence length to keep
- `DATASET_NAME="librispeech"` — affects subdirectory naming

### 2. Checkpoint / Resume System

Because each pipeline stage can take minutes to hours, `utils.sh` implements a file-based checkpoint system so that if the pipeline crashes or is interrupted, it can resume from exactly where it stopped rather than starting over.

The checkpoint file lives at:
```
$DATA_ROOT/checkpoints/librispeech/progress.checkpoint
```

Each line in the file records the state of one pipeline step:
```
create_manifests_train:IN_PROGRESS
create_manifests_train:COMPLETED
create_rVADfast:IN_PROGRESS
...
```

Three functions manage this:

**`is_completed(step)`** — checks whether a `step:COMPLETED` line exists. If yes, the step is skipped.

**`mark_in_progress(step)`** — writes `step:IN_PROGRESS` before a step begins. If the pipeline crashes mid-step, this marker remains so you know where it failed.

**`mark_completed(step)`** — writes `step:COMPLETED` after a step finishes successfully.

To force a step to re-run, delete its `step:COMPLETED` line from the checkpoint file.

---

## Stage 1: Manifest Creation

**Script:** `wav2vec_functions.sh` → `create_manifests_train`, `create_manifests_val`, `create_manifests_test`

**Tool:** `fairseq_/examples/wav2vec/wav2vec_manifest.py`

A **manifest** is a plain-text TSV file that acts as an index to the audio dataset. It has the following format:

```
/absolute/path/to/audio/root
relative/path/to/file1.wav   <num_frames>
relative/path/to/file2.wav   <num_frames>
...
```

The first line is the root directory. Each subsequent line is a relative path to one audio file plus its frame count (number of audio samples). Fairseq reads manifests instead of scanning directories directly — this is faster and keeps the data structure flexible.

`wav2vec_manifest.py` recursively scans the provided audio directory for `.wav` files, counts their frames, and writes the TSV.

Three manifests are created:
- `manifests/train.tsv` — training audio
- `manifests/valid.tsv` — validation audio
- `manifests_nonsil/test.tsv` — test audio (placed directly in the nonsil dir since test audio is not silence-stripped)

The `--valid-percent` flag controls the train/valid split. We pass `0` for train (100% goes to train) and `1.0` for val (100% goes to valid).

---

## Stage 2: Voice Activity Detection (VAD)

**Script:** `wav2vec_functions.sh` → `create_rVADfast`

**Tool:** `vads.py` + `rVADfast`

Raw audio contains silence, background noise, and non-speech segments. Training a GAN on silence teaches it nothing useful. This stage detects which parts of each audio file contain actual speech.

### How it works

`vads.py` reads the manifest from stdin (so it knows the root path and file list), processes each audio file through rVADfast, and writes a `.vads` file to stdout.

For each audio file, rVADfast performs:
1. **Spectral flatness** (`sflux`) — measures how noise-like vs. tonal a frame is. Speech has low spectral flatness (tonal); noise has high spectral flatness. Returns a frame-level array `ft` and the number of frames `n_frames`.
2. **Pitch block detection** (`pitch_block_detect`) — groups voiced frames into blocks, reinforcing the speech/non-speech boundary.
3. **High-energy noise removal** (`snre_highenergy`) — identifies and attenuates high-energy noise segments.
4. **VAD decision** (`snre_vad`) — produces a binary frame-level mask: `1` = speech, `0` = silence/noise.

The output `.vads` file has one line per audio file listing the speech segments as `start:end` sample pairs:
```
12800:45600 48000:102400 ...
```

A key code fix was required: the `sflux()` function in rVADfast originally returned only one value, but `vads.py` expected it to return `(s_flatness, n_frames)`. The `fixing_sflux()` function in `wav2vec_functions.sh` patches `speechproc.py` with `sed` before the step runs.

---

## Stage 3: Silence Removal

**Script:** `wav2vec_functions.sh` → `remove_silence`

**Tool:** `fairseq_/examples/wav2vec/unsupervised/scripts/remove_silence.py`

Using the `.vads` segment files from Stage 2, this step physically writes new `.wav` files containing only the speech portions of each recording. Silence segments are discarded.

Input: original `.wav` files + `.vads` segment files
Output: trimmed `.wav` files written to `$NONSIL_AUDIO/train/` and `$NONSIL_AUDIO/val/`

These trimmed files are shorter and contain only speech signal, which is what the GAN will learn from.

---

## Stage 4: Non-Silence Manifests

**Script:** `wav2vec_functions.sh` → `create_manifests_nonsil_train`, `create_manifests_nonsil_val`

Same process as Stage 1 but now pointing at the silence-free audio directories:
- `manifests_nonsil/train.tsv` → indexes `processed_audio/train/`
- `manifests_nonsil/valid.tsv` → indexes `processed_audio/val/`

Downstream stages (feature extraction, GAN training) read from these nonsil manifests.

---

## Stage 5: Audio Feature Preparation

**Script:** `wav2vec_functions.sh` → `prepare_audio`

**Tool:** `fairseq_/examples/wav2vec/unsupervised/scripts/prepare_audio.sh` (called via `zsh`)

This is the most computationally intensive stage. It converts raw audio waveforms into compact, informative feature vectors that the GAN can learn from. It runs four sub-steps:

### 5.1 — wav2vec 2.0 Feature Extraction

The pretrained `wav2vec_vox_new.pt` model processes each audio file and extracts representations from **layer 14** of the transformer (empirically the most phonemically informative layer). Each 20ms frame of audio becomes a 1024-dimensional vector.

Output: `.npy` files containing raw feature matrices, one per split (train/valid/test).

### 5.2 — PCA Dimensionality Reduction

The 1024-dimensional raw features are projected down to **512 dimensions** using Principal Component Analysis (PCA). PCA is fit on a random sample of training frames (controlled by `NEW_SAMPLE_PCT=0.5`, meaning 50% of training audio) for speed, then applied to all splits.

Output: `clustering/librispeech/train.npy`, `valid.npy`, `test.npy` at 512 dims.

### 5.3 — K-Means Clustering (128 clusters)

K-means is run on the PCA-reduced training features to learn **128 cluster centroids**. Each centroid represents a pseudo-phoneme — a recurring acoustic pattern in the data. The number 128 was chosen to be larger than the true phoneme inventory (~40 for English) so the model has flexibility.

The clustering spec is `CLUS128` (no additional PCA inside faiss, 128 clusters). The trained centroids are saved as `centroids.npy`.

Then each frame in every split is assigned to its nearest centroid, producing a sequence of discrete cluster IDs (integers 0–127) per file.

Output: `clustering/librispeech/CLUS128/` containing centroids and per-split cluster ID files.

### 5.4 — Mean Pooling

Consecutive frames with the same cluster ID are collapsed into single tokens by mean-pooling their feature vectors. This removes repetition (if a phoneme spans 5 frames, those 5 frames become 1 averaged frame) and dramatically shortens the sequences. This step converts frame-level features into segment-level features aligned with the pseudo-phoneme boundaries.

Output: `clustering/librispeech/precompute_pca512_cls128_mean_pooled/` — the final feature directory consumed by the GAN.

The `update_sample_pct()` and `update_batch_size()` helper functions patch `prepare_audio.sh` in-place using `sed` before it runs, injecting the values from `utils.sh`.

---

## Stage 6: Text Preparation

**Script:** `wav2vec_functions.sh` → `prepare_text`

**Tool:** `fairseq_/examples/wav2vec/unsupervised/scripts/prepare_text.sh` (called via `zsh`)

The GAN needs "real" samples — phoneme sequences from natural language. These come from an unpaired text corpus (no audio required). This stage converts raw text into phoneme sequences and trains a language model on them.

Sub-steps:
1. **Language identification** — FastText (`lid.176.bin`) filters out any non-target-language sentences from the corpus. The `0.25` threshold means sentences with ≥ 25% confidence of being the target language are kept.
2. **G2P phonemization** — The G2P (Grapheme-to-Phoneme) tool converts each word into a sequence of phoneme symbols (e.g. "hello" → `HH AH0 L OW1`). Sequences shorter than `MIN_PHONES=3` are discarded.
3. **KenLM language model** — A 4-gram language model is trained on the phoneme sequences using KenLM. This LM is used during Viterbi decoding to bias transcriptions toward phonotactically plausible sequences.

A code fix is applied before this step: `replace_std_endl()` patches `add-self-loop-simple.cc` in the Fairseq Kaldi integration, replacing `std::endl` with `"\n"` for compatibility with the pykaldi build.

Output (in `$TEXT_OUTPUT/phones/`):
- `phones.txt` — all phonemized sequences (one per line)
- `lm.phones.filtered.04.bin` — compiled KenLM binary (used at decoding time)
- `dict.txt` — phoneme vocabulary

---

## Stage 7: GAN Training

**Script:** `gans_functions.sh` → `train_gans`

**Tool:** `fairseq-hydra-train` (Fairseq's Hydra-based training launcher)

**Config:** `fairseq_/examples/wav2vec/unsupervised/config/gan/w2vu.yaml`

This is where the three classes from `wav2vec_u_gan.py` come to life as a full training loop.

### What the GAN learns

The GAN frames speech recognition as an **adversarial game**:

- The **Generator** takes mean-pooled audio features (512-dim segment vectors from Stage 5) and maps them to a distribution over phonemes at each time step. It tries to produce sequences that look like real phoneme sequences.
- The **Discriminator** sees either real phoneme sequences (from the text corpus, Stage 6) or generated sequences and tries to tell them apart.
- The **Generator** improves by fooling the Discriminator; the **Discriminator** improves by catching the Generator.

Over many iterations, the Generator is forced to produce phoneme sequences that match the statistical distribution of real speech — which is exactly what transcription is.

### The three GAN components

**`RealData`**
- Loads `phones.txt` (from Stage 6) into memory
- Randomly samples mini-batches of phoneme sequences
- Converts them to one-hot vectors `[B, T, vocab_size]`
- These are the "real" samples the Discriminator is trained to recognise

**`Generator`**
- Input: `[B, T, 512]` — mean-pooled audio features
- Architecture: BatchNorm → Dropout → causal Conv1d (kernel=4)
- Output: `dense_x` (softmax probabilities), `token_x` (argmax one-hot), `logits` (raw)
- Causal padding ensures each position only attends to past frames (left-to-right, like speech)
- `dense_x` is differentiable and fed to the Discriminator during generator updates
- `token_x` is the discrete hard prediction used for evaluation

**`Discriminator`**
- Input: `[B, T, vocab_size]` — either real one-hot or generated softmax vectors
- Architecture: Linear projection → 2× (causal Conv1d + GELU + Dropout) with residual connections → mean pool → scalar score
- Output: a single real/fake logit per sequence
- Causal convolutions mean it judges sequences left-to-right, capturing temporal phoneme patterns

### Loss functions

Four losses are combined during training:

| Loss | Formula | Purpose |
|---|---|---|
| `adversarial_loss` | BCE on real/fake scores | Core GAN objective |
| `gradient_penalty` | `E[(‖∇D(interpolated)‖ − 1)²]` | WGAN-GP: stabilises training, prevents collapse |
| `smoothness_loss` | MSE between consecutive logit frames | Prevents jerky, unrealistic phoneme jumps |
| `code_penalty` | `(V − exp(H(p̄))) / V` | Penalises the Generator for using only a few phonemes |

Training hyperparameters (set in the `fairseq-hydra-train` call):
- Generator learning rate: `0.00004`
- Discriminator learning rate: `0.00002`
- `code_penalty` weight: `6` and `10` (swept across seeds)
- `gradient_penalty` weight: `0.5` and `1.0` (swept)
- `smoothness_weight`: `1.5`
- 5 random seeds run in parallel via Hydra multirun

Checkpoints are saved under `multirun/<date>/<time>/checkpoint_best.pt`.

---

## Stage 8: Evaluation — Viterbi Decoding

**Script:** `eval_functions.sh` → `transcription_gans_viterbi`

**Tool:** `fairseq_/examples/wav2vec/unsupervised/w2vu_generate.py`

**Config:** `fairseq_/examples/wav2vec/unsupervised/config/generate/viterbi`

After training, the Generator is used to transcribe audio. The Viterbi algorithm finds the most probable phoneme sequence given:
1. The Generator's per-frame phoneme probability distributions (the acoustic model)
2. The KenLM 4-gram language model (the language model)

This is a classic HMM-style decoding: at each frame, it balances the acoustic evidence with how likely that phoneme sequence is in natural language.

Input:
- Trained model checkpoint (`.pt` file, passed as argument to `run_eval.sh`)
- Mean-pooled audio features (`precompute_pca512_cls128_mean_pooled/`)
- KenLM binary (`lm.phones.filtered.04.bin`)

Output:
- `data/transcription_phones/test.txt` — one phoneme sequence per line, the final transcription of each test utterance

---

## Full Data Flow Diagram

```
Raw .wav files
      │
      ▼
[Stage 1] wav2vec_manifest.py
      │  Scan audio dirs, count frames
      ▼
manifests/train.tsv, valid.tsv
      │
      ▼
[Stage 2] vads.py + rVADfast
      │  Frame-level speech/silence detection
      ▼
manifests/train.vads, valid.vads
      │
      ▼
[Stage 3] remove_silence.py
      │  Write speech-only .wav files
      ▼
processed_audio/train/, processed_audio/val/
      │
      ▼
[Stage 4] wav2vec_manifest.py (again)
      │  Index silence-free audio
      ▼
manifests_nonsil/train.tsv, valid.tsv
      │
      ▼
[Stage 5] prepare_audio.sh
      │  wav2vec feature extraction (layer 14, 1024-dim)
      │  PCA → 512-dim
      │  K-means → 128 pseudo-phoneme cluster IDs
      │  Mean pool consecutive same-cluster frames
      ▼
clustering/librispeech/precompute_pca512_cls128_mean_pooled/
      │                                    │
      │                                    │
Unpaired text corpus                       │
      │                                    │
      ▼                                    │
[Stage 6] prepare_text.sh                 │
      │  Language ID filter (FastText)     │
      │  G2P phonemization                 │
      │  KenLM 4-gram LM training          │
      ▼                                    │
data/text/phones/                          │
  phones.txt  ◄──────── RealData ──────────┤
  lm.phones.filtered.04.bin                │
      │                                    │
      └──────────────┬─────────────────────┘
                     │
                     ▼
[Stage 7] fairseq-hydra-train (w2vu.yaml)
      │  Generator: audio features → phoneme probs
      │  Discriminator: real vs fake phoneme sequences
      │  Losses: adversarial + WGAN-GP + smoothness + code penalty
      ▼
multirun/<date>/<time>/checkpoint_best.pt
      │
      ▼
[Stage 8] w2vu_generate.py (Viterbi)
      │  Generator acoustic probs + KenLM LM
      ▼
data/transcription_phones/test.txt
(final phoneme transcriptions)
```

---

## Orchestration Scripts

### `run_pipeline.sh`
Entry point added to avoid shell argument-wrapping bugs. Hardcodes the dataset paths as variables and calls `run_wav2vec.sh`. Run as:
```bash
./run_pipeline.sh
```

### `run_wav2vec.sh`
Sources `wav2vec_functions.sh` and calls the data prep stages in order:
```
create_dirs → activate_venv → setup_path
→ create_manifests_train/val/test
→ create_rVADfast → remove_silence
→ create_manifests_nonsil_train/val
→ prepare_audio → prepare_text
```

### `run_gans.sh`
Sources `gans_functions.sh` and calls `train_gans`.

### `run_eval.sh`
Sources `eval_functions.sh` and calls `transcription_gans_viterbi`. Takes the checkpoint path as argument:
```bash
./run_eval.sh "multirun/<date>/<time>/checkpoint_best.pt"
```
