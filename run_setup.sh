#!/bin/bash

# Source the implementation file to load all functions
source "$(dirname "$0")/setup_functions.sh"


# Calling functions defined in setup_functions.sh
log "Starting custom setup sequence..."

# ===============================================
# 1. SETUP AND PREREQUISITES
# ===============================================

basic_dependencies
# Install essential system packages and libraries required for compilation 

#the if statements, checks for the presence of a gpu.
if lspci | grep -iq "nvidia"; then
    echo "NVIDIA GPU detected. Proceeding with GPU setup..."

    # cuda_installation
    # Install the NVIDIA CUDA Toolkit. This provides the compiler (nvcc) and
    # libraries needed for GPU acceleration of deep learning frameworks.

    # nvidia_drivers_installation
    # gpu_drivers_installation
    # Install or update the proprietary NVIDIA GPU drivers to ensure hardware 
    # is correctly recognized and accessible by CUDA.
else
    echo "No NVIDIA GPU found. Using CPU-only mode."
fi

create_dirs
# Create necessary directory structures for storing data, checkpoints, and final output 

# ===============================================
# 2. PYTHON ENVIRONMENT AND CORE FRAMEWORKS
# ===============================================

setup_venv
# Create and activate a isolated Python virtual environment 
# to manage project-specific dependencies without 
# conflicting with the system Python installation.

install_pytorch_and_other_packages
# Install the PyTorch deep learning framework, which is the foundational 
# tensor library used for training and running models. This often 
# requires selecting a version compatible with the installed CUDA version.
#also install other packapges need for a smooth execution of unsupervised wav2vec U. 

install_fairseq
# Install the Fairseq sequence modeling toolkit (developed by Facebook AI). 
# This library is typically used to implement and train state-of-the-art 
# models like Wav2Vec 2.0 and Transformer networks.

install_flashlight #installs flashlight-text and flashlight-sequence
# This is often used for fast decoding in speech recognition models.

# ===============================================
# 3. DOMAIN-SPECIFIC TOOLS
# ===============================================

install_kenlm
# Install the KenLM language modeling toolkit. This is crucial for 
# integrating a language model during the decoding phase of speech 
# recognition to improve transcription accuracy.

install_rVADfast
# Install a fast implementation of Robust Voice Activity Detection (rVAD). 
# This tool is used to identify and segment speech portions within 
# audio files, filtering out silence and noise.
# ===============================================
# 4. DATA AND MODEL ARTIFACTS
# ===============================================

download_pretrained_model
# Download the wav2vec_vox_new.pt pre-trained deep learning model checkpoint

download_languageIdentification_model
# Download the lid.176.bin pre-trained model specifically designed for determining the 
# language spoken in an audio file.

log "Custom setup sequence completed!"
