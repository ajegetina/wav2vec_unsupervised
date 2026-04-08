#!/bin/bash
TRAIN="/home/ajegetina/coding/school/ashesi/semester-2/nlp/prosits/prosit-3/data/LibriSpeech/small/train"
VAL="/home/ajegetina/coding/school/ashesi/semester-2/nlp/prosits/prosit-3/data/LibriSpeech/small/val"
TEST="/home/ajegetina/coding/school/ashesi/semester-2/nlp/prosits/prosit-3/data/LibriSpeech/small/test"
TEXT="/home/ajegetina/coding/school/ashesi/semester-2/nlp/prosits/prosit-3/data/LibriSpeech/text_corpus.txt"

cd "$(dirname "$0")"
./run_wav2vec.sh "$TRAIN" "$VAL" "$TEST" "$TEXT" 2>&1 | tee wav2vec_run.log
