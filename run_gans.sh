#!/bin/bash

# Source the function definitions
source "$(dirname "$0")/gans_functions.sh"
activate_venv  
setup_path 
create_dirs #creates directories for storing outputs from the different steps 

train_gans

log "Pipeline completed successfully!"
