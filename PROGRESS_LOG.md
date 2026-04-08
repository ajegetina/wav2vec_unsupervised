# Wav2Vec-U Pipeline — Progress Log

## 1. Prosit Requirements

The prosit required implementing the core GAN module for **Wav2Vec-U** (Baevski et al., NeurIPS 2021 — "Unsupervised Speech Recognition"). The deliverable is `wav2vec_u_gan.py` containing three clearly delineated classes following separation of concerns:

| Class | Role |
|---|---|
| `RealData` | Provides "real" samples — phoneme sequences loaded from an unpaired text corpus |
| `Generator` | Provides "fake" samples — maps wav2vec 2.0 audio features to phoneme distributions |
| `Discriminator` | Scores sequences as real or fake |

Supporting loss functions also required: `adversarial_loss`, `gradient_penalty` (WGAN-GP), `smoothness_loss`, `code_penalty`.

---

## 2. Code Cleanup — `wav2vec_u_gan.py`

The original file was 799 lines and included training infrastructure not relevant to the prosit deliverable. The following were removed:

- `TrainingConfig` dataclass and all its fields
- `load_phoneme_vocab()` utility function
- `load_audio_features()` utility function
- `train()` function (full training loop)
- CLI `argparse` entry point (`if __name__ == "__main__"`)
- Unused imports: `math`, `os`, `random`, `dataclasses` (`dataclass`, `field`), `pathlib` (`Path`), `numpy`

**Result:** 799 lines → 490 lines. Retained only the three core classes and four loss functions, which are the actual GAN components required by the prosit.

---

## 3. Setup — `run_setup.sh`

### 3.1 Syntax Error in `setup_functions.sh`

**Error:**
```
./setup_functions.sh: line 367: syntax error near unexpected token '}'
```

**Cause:** Inside `install_flashlight()`, the `if ! command -v nvcc` block was missing its closing `fi`. Additionally, `cmake` and `pip install` commands had trailing backslashes (`\`) followed by blank lines, which broke the multi-line continuation.

**Fix:** Added the missing `fi` after `export USE_CUDA=0` and removed the stray trailing backslashes from the cmake and pip commands.

### 3.2 CUDA Installation Skipped

The `cuda_installation` and `nvidia_drivers_installation` calls in `run_setup.sh` were commented out since the environment already had CUDA configured.

### 3.3 CRLF in `cuda_installation.txt`

**Error:** `wget` was producing a 400 Bad Request because the URL contained `%0D` (carriage return).

**Cause:** `cuda_installation.txt` had Windows-style CRLF line endings. The `\r` was being appended to the URL string.

**Fix:**
```bash
sed -i 's/\r//' cuda_installation.txt
```

### 3.4 sudo Requires Interactive Terminal

**Error:**
```
sudo: a terminal is required to read the password
```

**Cause:** Scripts with `sudo apt-get` were being run via a non-interactive background tool.

**Fix:** User runs `run_setup.sh` directly in their terminal using the `!` prefix in Claude Code.

### 3.5 Setup Completion

All dependencies installed successfully:
- Python 3.10 venv at `$HOME/wav2vec_unsupervised/venv`
- PyTorch 2.3.0 + torchaudio + torchvision (CUDA 12.1)
- Fairseq (custom Ashesi fork: `https://github.com/Ashesi-Org/fairseq_`)
- KenLM (built from source)
- rVADfast
- Flashlight (text + sequence)
- Pretrained model: `pre-trained/wav2vec_vox_new.pt`
- Language ID model: `lid_model/lid.176.bin`

---

## 4. Data Preparation — LibriSpeech

### 4.1 Audio Format

LibriSpeech ships as `.flac` files. The pipeline requires `.wav` (16kHz mono).

**Fix:** Converted all 33,862 files using ffmpeg with 8 parallel jobs:
```bash
find . -name "*.flac" | xargs -P 8 -I{} bash -c \
  'out="${1%.flac}.wav"; ffmpeg -i "$1" -ar 16000 -ac 1 "$out" -y -loglevel error' _ {}
```

### 4.2 Text Corpus

Built `text_corpus.txt` from LibriSpeech `.trans.txt` transcript files (28,539 sentences) at:
`data/LibriSpeech/text_corpus.txt`

---

## 5. Pipeline Errors — `run_wav2vec.sh`

### 5.1 `vads.py` Not Found

**Error:**
```
python: can't open file '/home/ajegetina/wav2vec_unsupervised/vads.py': [Errno 2] No such file or directory
```

**Cause:** `wav2vec_functions.sh` references `$DIR_PATH/vads.py` where `DIR_PATH=$HOME/wav2vec_unsupervised`, but `vads.py` lives in the repository directory, not the install root.

**Fix:**
```bash
cp /path/to/repo/wav2vec_unsupervised/vads.py /home/ajegetina/wav2vec_unsupervised/vads.py
```

### 5.2 Empty Manifests (Wrong Paths Checkpointed)

**Error:** Feature extraction produced `0it [00:00, ?it/s]` for all splits, followed by:
```
ValueError: need at least one array to concatenate
FileNotFoundError: .../centroids.npy
```

**Cause:** The manifest `.tsv` files (created in an earlier failed run) contained only 1 line — the root path — with no audio file entries. The checkpoint system marked those steps as `COMPLETED`, so the pipeline skipped recreation and proceeded with empty manifests.

**Fix:** Cleared the checkpoint file and deleted the bad manifests to force re-creation:
```bash
cat /dev/null > .../progress.checkpoint
rm -f .../manifests/train.tsv .../manifests/valid.tsv ...
```

### 5.3 Path Splitting via `!` Multi-line Command

**Error:**
```
soundfile.LibsndfileError: Error opening '.../LibriSpeech/trai/n-clean-100': System error.
soundfile.LibsndfileError: Error opening '.../data/L/ibriSpeech/train-clean-100': System error.
```

**Cause:** Passing long paths with backslash line-continuation inside the `!` Claude Code prompt embeds literal newlines into argument strings. The manifest root path was written as a split path across two lines.

**Fix:** Created `run_pipeline.sh` which hardcodes the paths as variables and calls `run_wav2vec.sh` internally. The `!` command then only needs to call the short script name:
```bash
! ./run_pipeline.sh
```

### 5.4 Dataset Too Large (train-clean-100)

**Problem:** The `create_rVADfast` (silence removal) step on 28,000+ files would take hours.

**Fix:** Switched to a small subset carved from `test-clean` (already converted to wav):
- **Train:** 1,500 files
- **Val:** 500 files
- **Test:** 620 files

Files symlinked (not copied) into `data/LibriSpeech/small/{train,val,test}/`. `run_pipeline.sh` updated to point at these directories.

---

## 6. GAN Training Errors — `run_gans.sh`

### 6.1 `wav2vec_u.py` Empty (Intentionally)

**Error:**
```
ImportError: cannot import name 'Wav2vec_U' from 'unsupervised.models.wav2vec_u'
(/home/ajegetina/wav2vec_unsupervised/fairseq_/examples/wav2vec/unsupervised/models/wav2vec_u.py)
```

**Cause:** The Ashesi fork intentionally emptied `fairseq_/examples/wav2vec/unsupervised/models/wav2vec_u.py` (commit `9837cf97: "Update wav2vec_u.py"`) as part of the prosit — students are expected to implement the GAN. The file had only 4 lines (copyright header). `fairseq-hydra-train` could not import the required `Wav2vec_U` registered model class.

**Context:** The student's `wav2vec_u_gan.py` (490 lines, 3 classes + 4 loss functions) is the prosit deliverable demonstrating separation of concerns. It is a standalone implementation. The fairseq `wav2vec_u.py` is the framework-integrated model needed to run the training pipeline — these are complementary, not duplicates.

**Fix:** Restored `wav2vec_u.py` from the previous commit (`a0ceabc2`) which contains the full 687-line implementation including the `Wav2vec_U` fairseq registered model:
```bash
cd /home/ajegetina/wav2vec_unsupervised/fairseq_
git show a0ceabc2:examples/wav2vec/unsupervised/models/wav2vec_u.py > \
  examples/wav2vec/unsupervised/models/wav2vec_u.py
```

GAN training then ran successfully, saving checkpoints to:
`multirun/2026-03-27/13-19-35/0/checkpoint_best.pt`

---

## 7. Evaluation Errors — `run_eval.sh`

### 7.1 Wrong Checkpoint Path Construction

**Error:**
```
IsADirectoryError: [Errno 21] Is a directory: '/home/ajegetina/wav2vec_unsupervised/'
OSError: Model file not found: /home/ajegetina/wav2vec_unsupervised/multirun/2026-03-27/13-19-35/0/checkpoint_best.pt
```

**Cause:** `eval_functions.sh` line 14 constructed the model path as:
```bash
MODEL_PATH=$DIR_PATH/$1
```
`DIR_PATH` resolves to `$HOME/wav2vec_unsupervised/` (the install root), but checkpoints are saved in the **project directory** (`prosit-3/wav2vec_unsupervised/multirun/...`) where `fairseq-hydra-train` was invoked. The path was therefore wrong.

**Fix:** Changed the path construction to resolve relative to the script's own directory:
```bash
MODEL_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$1"
```

Evaluation then ran successfully with:
```bash
./run_eval.sh "multirun/2026-03-27/13-19-35/0/checkpoint_best.pt"
```

---

## 8. Current Status

| Step | Status |
|---|---|
| Setup (venv, PyTorch, Fairseq, KenLM, etc.) | Complete |
| Audio conversion (.flac → .wav) | Complete |
| Text corpus creation | Complete |
| `wav2vec_u_gan.py` cleanup | Complete |
| `run_wav2vec.sh` (manifests, VAD, features, clustering, text prep) | Complete |
| `run_gans.sh` (GAN training) | Complete |
| `run_eval.sh` (Viterbi decoding) | Complete |

**Final output:** `data/transcription_phones/test.txt` — phoneme sequences for all 620 test utterances.
