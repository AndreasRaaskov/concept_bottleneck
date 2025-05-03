#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J Independent
### -- ask for number of cores (default: 1) --
#BSUB -n 4 
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 12:00
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

<<<<<<< HEAD:HPC/train_independent.sh
python3 main.py mode=Independent
=======
#python3 main.py mode=Sequential experiment_name=Sequential_NoM_NoPrtraining pretrained=False
#python3 main.py mode=Sequential experiment_name=Sequential_NoM_NoPrtraining_Rezise pretrained=False transform_method=resize
#python3 main.py mode=Sequential experiment_name=Sequential_NoM
#python3 main.py mode=Sequential experiment_name=Sequential_MV_weigthted CNN_epochs=1000 CUB_dataloader.use_majority_voting=True
#python3 main.py mode=Sequential experiment_name=Sequential_NoMV_weigthted CNN_epochs=1000 CUB_dataloader.use_majority_voting=False

python3 main.py mode=Sequential XtoC_path=Sequential_NoMV_weigthted/best_XtoC_model.pth CtoY_path=NoMajority_models/Concepts_CKTP_NoPretraining/best_XtoC_model.pth output_dir=Concepts_CKTP_NoPretraining
python3 main.py mode=Sequential XtoC_path=Sequential_NoMV_weigthted/best_XtoC_model.pth CtoY_path=NoMajority_models/Concepts_CKTP_Resize/best_XtoC_model.pth output_dir=Concepts_CKTP__Resize
>>>>>>> d8ea00c (Make epochelengt of sequential training custermeisble):HPC/train_seqential.sh
