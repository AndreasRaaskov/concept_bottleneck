"""
NOTE this script is not yet finisehed
Script that takes a model and a dataset and evaluates the model on the dataset.
"""

import torch
import hydra
import os
from omegaconf import DictConfig,OmegaConf
from data_loaders import CUB_extnded_dataset
from models import get_inception_transform
from utils.analysis import TrainingLogger
from sailency import get_saliency_maps,saliency_score_image,get_visible_consepts
import tqdm


def main(cfg: DictConfig):

    #Make the analysis object
    logger = TrainingLogger(os.path.join(cfg.output_dir, 'Evaluation.json'))

    if cfg.device.lower() == "auto":
        device= "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
    else:
        device = cfg.device
    
    
    if cfg.mode == "Joint":
        #Load the model
        model = torch.load(cfg.joint_path, map_location=torch.device(device))
        model.eval()
        model.use_sigmoid = True #line to stay compatible with the old models
        model.to(device)


    elif cfg.mode == "Independent" or cfg.mode == "Sequential":
        # Load x to c model
        XtoC_model = torch.load(cfg.XtoC_path, map_location=torch.device(device))
        XtoC_model.eval()
        XtoC_model.use_sigmoid = True
        XtoC_model.to(device)

        # Load c to y model
        CtoY_model = torch.load(cfg.CtoY_path, map_location=torch.device(device))
        CtoY_model.eval()
    
    elif cfg.mode == "Standard":

        #Load the model
        model = torch.load(cfg.model_path, map_location=torch.device(device))
        model.eval()
        model.to(device)
    else:
        raise NotImplementedError(f"Model type {cfg.model_type} is not implemented")


    #Make the dataset
    transform = get_inception_transform(mode=cfg.split, methode= cfg.transform_method)
    data_set = CUB_extnded_dataset(mode=cfg.split,config_dict=cfg.CUB_dataloader,transform=transform)

    #Make a mask if a model need to be tested on another dataset than it was trained on
    if cfg.CUB_mask.use:
        mask = CUB_extnded_dataset(mode=cfg.split,config_dict=cfg.CUB_mask,transform=transform).concept_mask

    # Calculate accuracy
    for i in tqdm.tqdm(range(len(data_set))):

        X, C, Y , _ = data_set[i]

        if cfg.CUB_mask.use:
            #Apply the mask
            C = C[mask]

        #Unsqueese the data and move it to the device
        X = X.unsqueeze(0).to(device)
        C = C.unsqueeze(0).to(device)
        Y = Y.unsqueeze(0).to(device)

        if cfg.mode == "Standard":
            #Forward pass
            Y_hat = model(X)
        
        else: # Evaluate both C and Y
            if cfg.mode == "Joint":
                #Forward pass
                C_hat, Y_hat = model(X)
            
            elif cfg.mode == "Independent" or cfg.mode == "Sequential":
                #Forward pass
                C_hat = XtoC_model(X)
                Y_hat = CtoY_model(C_hat)

            if cfg.CUB_mask.use:
                #Only evaluated masked concepts
                C_hat = C_hat[..., mask]


            #Update concept accuracy
            logger.update_concept_accuracy(mode="test", predictions=C_hat, ground_truth=C)

        #Update the class logger
        logger.update_class_accuracy(mode="test",logits=Y_hat, correct_label=Y)

    if cfg.sailency == True and cfg.mode != "Standard":

        for i in tqdm.tqdm(range(len(data_set))):

            X, C, Y , coordinates = data_set[i]

            if cfg.CUB_mask.use:
                #Apply the mask
                C = C[mask]
                coordinates = [coordinates[i] for i in mask]

            concept_list,coordinates = get_visible_consepts(coordinates)

            sailency_maps = get_saliency_maps(X,concept_list,model,method_type=cfg.sailency_methode)

            sailency_score = saliency_score_image(sailency_maps,coordinates=coordinates)

            logger.update_sailency_score(mode="test",score=sailency_score)

    logger.log_metrics(i)
        

        



def XtoC_test(model,device,cfg: DictConfig):
    """
    Test a model on the X to C data and make sailency maps for all concepts.
    
    args:
    model 
    """


    #Make the dataset
    transform = get_inception_transform(mode=cfg.split, methode= cfg.transform_method)
    data_set = CUB_extnded_dataset(mode=cfg.split,config_dict=cfg.CUB_dataloader,transform=transform)

    #Make a mask if a model need to be tested on another dataset than it was trained on
    if cfg.CUB_mask.use:
        mask = CUB_extnded_dataset(mode=cfg.split,config_dict=cfg.CUB_mask,transform=transform).concept_mask

    


    #Make the analysis object
    logger = TrainingLogger(os.path.join(cfg.output_dir, 'XtoC_test.json'))

    for i in tqdm.tqdm(range(len(data_set))):
        

        X, C, Y , coordinates = data_set[i]

        if cfg.CUB_mask.use:
            #Apply the mask
            C = C[mask]
            coordinates = [coordinates[i] for i in mask]


        #Unsqueese the data and move it to the device
        X = X.unsqueeze(0).to(device)
        C = C.unsqueeze(0).to(device)
        Y = Y.unsqueeze(0).to(device)

        #Forward pass
        C_hat = model(X)

        if cfg.CUB_mask.use:
            #Only evaluated masked concepts
            C_hat = C_hat[..., mask]

    
        #Update the logger
        #logger.update_class_accuracy(mode="test",logits=Y_hat, correct_label=Y)
        logger.update_concept_accuracy(mode="test", predictions=C_hat, ground_truth=C)

        concept_list,coordinates = get_visible_consepts(coordinates)

        sailency_maps = get_saliency_maps(X,concept_list,model,method_type="vanilla")

        sailency_score = saliency_score_image(sailency_maps,coordinates=coordinates)

        logger.update_sailency_score(mode="test",score=sailency_score)

    logger.log_metrics(i)

def CtoY_test(model,cfg: DictConfig):
    pass

def XtoY_test(model,cfg):
    pass
        
if __name__ == '__main__':
    cofig_dict = OmegaConf.load('config/evaluation.yaml')
    main(cofig_dict)