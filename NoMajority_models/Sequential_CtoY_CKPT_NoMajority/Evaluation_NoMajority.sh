#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J EvaluateNoMajority
### -- ask for number of cores (default: 1) --
#BSUB -n 4 
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 8:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o HPC/systemoutput/gpu_%J.out
#BSUB -e HPC/systemoutput/gpu_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/11.6

/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

source env/bin/activate


# Evaluate all NoMajority models

# Joint
#python3 evaluation.py mode=Joint joint_path=NoMajority_models/Joint_CKPT_NoMajority/best_Joint_model.pth output_dir=NoMajority_models/Joint_CKPT_NoMajority

# Independent
#python3 evaluation.py mode=Independent X_to_C_path=NoMajority_models/Concepts_CKPT_NoMajority/best_XtoC_model.pth CtoY_path=NoMajority_models/Independent_CtoY_CKPT_NoMajority/best_CtoY_model.pth output_dir=NoMajority_models/Independent_CtoY_CKPT_NoMajority

# Sequential
#python3 evaluation.py mode=Sequential X_to_C_path=NoMajority_models/Concepts_CKPT_NoMajority/best_XtoC_model.pth CtoY_path=NoMajority_models/Sequntial_NoMajority_end/best_CtoY_model.pth output_dir=NoMajority_models/Sequential_CtoY_CKPT_NoMajority


# Evaluate NoPretraining models
# Joint
python3 evaluation.py mode=Joint joint_path=NoMajority_models/Joint_CKTP_NoPretraining/best_Joint_model.pth output_dir=NoMajority_models/Joint_CKTP_NoPretraining

# Independent
python3 evaluation.py mode=Independent X_to_C_path=NoMajority_models/Concepts_CKTP_NoPretraining/best_XtoC_model.pth CtoY_path=NoMajority_models/Independent_CtoY_CKPT_NoMajority/best_CtoY_model.pth output_dir=NoMajority_models/Independent_NoPretraining

# Sequential
python3 evaluation.py mode=Sequential X_to_C_path=NoMajority_models/Concepts_CKPT_NoMajority/best_XtoC_model.pth CtoY_path=Sequential_CKTP__NoPretraining/best_CtoY_model.pth output_dir=Sequential_CKTP__NoPretraining

# Evaluate Resized models
# Joint
python3 evaluation.py mode=Joint joint_path=NoMajority_models/Joint_CKTP_resize/best_Joint_model.pth output_dir=NoMajority_models/Joint_CKTP_resize

# Independent
python3 evaluation.py mode=Independent X_to_C_path=NoMajority_models/Concepts_CKTP_Resize/best_XtoC_model.pth CtoY_path=NoMajority_models/Independent_CtoY_CKPT_NoMajority/best_CtoY_model.pth output_dir=NoMajority_models/Independent_Rezise

# Sequential
python3 evaluation.py mode=Sequential X_to_C_path=NoMajority_models/Concepts_CKPT_NoMajority/best_XtoC_model.pth CtoY_path=Sequential_CKTP__Resize/best_CtoY_model.pth output_dir=Sequential_CKTP__Resize




