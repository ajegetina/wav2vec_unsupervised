#!/bin/bash

# Source the function definitions

source "$(dirname "$0")/eval_functions.sh"

create_dirs
activate_venv 
#the trained checkpoints from train_gans will be stored in a folder called multirun. The checkpoint will be stored in this format 
#multirun --
 #         |
 #         day/month/year --
 #                         |
 #                         time --
 #                                |
 #                                checkpoint_best.pt
 #                                 checkpoint_last.pt
 #therefore it is advisable to manually provide the path to the exact checkpoint to use under the variable $CHECKPOINT_DIR  in the run_wav2vec.sh script
 

#Transcriptions from the GAN model 
#     transcription_gans_viterbi: outputs phonetic transcription in variable name $GANS_OUTPUT_PHONES

    transcription_gans_viterbi  #for these we need both train and validation since the train will be used by the HMM
