"""
NOTE this script is not yet finisehed
Script that takes a model and a dataset and evaluates the model on the dataset.
"""

import torch
import hydra
import os
from omegaconf import DictConfig,OmegaConf
from data_loaders import CUB_extnded_dataset,CUB_CtoY_dataset
from models import get_inception_transform
from utils.analysis import Logger
from sailency import get_saliency_maps,saliency_score_image,get_visible_consepts
import tqdm


def main(cfg: DictConfig):


    if cfg.device.lower() == "auto":
        device= "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
    else:
        device = cfg.device
    
    
    if cfg.mode == "Joint":
        #Load the model
        joint_model = torch.load(cfg.joint_path, map_location=torch.device(device))

        if cfg.original_model:
            XtoC_model = joint_model.first_model
            CtoY_model = joint_model.sec_model
        else:
            XtoC_model = joint_model.XtoC_model
            CtoY_model = joint_model.CtoY_model

        XtoC_model.eval()
        CtoY_model.eval()

        XtoC_model.to(device)
        CtoY_model.to(device)


    elif cfg.mode == "Independent" or cfg.mode == "Sequential":
        # Load x to c model
        XtoC_model = torch.load(cfg.XtoC_path, map_location=torch.device(device))
        XtoC_model.eval()
        XtoC_model.to(device)

        # Load c to y model
        CtoY_model = torch.load(cfg.CtoY_path, map_location=torch.device(device))
        CtoY_model.eval()

    
    elif cfg.mode == "Standard":

        #Load the model
        XtoY_model = torch.load(cfg.XtoY_path, map_location=torch.device(device))
        XtoY_model.eval()
        XtoY_model.to(device)
    else:
        raise NotImplementedError(f"Model type {cfg.model_type} is not implemented")


    #Make the dataset
    transform = get_inception_transform(mode=cfg.split, methode= cfg.transform_method)
    Non_majority_data_set = CUB_extnded_dataset(mode=cfg.split,config_dict=cfg.CUB_NonMajority_dataloader,transform=transform)

    concepts_name = Non_majority_data_set.consept_labels_names

    #Make the majority dataset and find the mask, we use CtoY dataset so we only need to loade the image ones 
    Majority_dataset = CUB_CtoY_dataset(mode=cfg.split,config_dict=cfg.CUB_majority_dataloader,transform=transform)
    mask = Majority_dataset.concept_mask
    concepts_name = concepts_name[mask]

    #Make the analysis object
    eval_logger = Logger(cfg=cfg,concept_mask=mask,concept_names=concepts_name)
    
    
    # Calculate accuracy
    for i in tqdm.tqdm(range(len(Non_majority_data_set))):

        #Get the data
        X, C_NoMajority, Y, _ = Non_majority_data_set[i]
        C_Majority, _= Majority_dataset[i]

        #Apply the mask to non majority voted concepts
        C_noMajority = C_NoMajority[mask]

        #Unsqueese the data and move it to the device
        X = X.unsqueeze(0).to(device)
        Y = Y.unsqueeze(0).to(device)

        C_noMajority = C_noMajority.unsqueeze(0).to(device)
        C_Majority = C_Majority.unsqueeze(0).to(device)


        if cfg.mode == "Standard":
            #Forward pass
            if cfg.original_model: #The original model returns a list where the first element is the y_hat
                Y_hat = torch.softmax(XtoY_model(X)[0],dim=1)
            else:
                Y_hat = torch.softmax(XtoY_model(X),dim=1)
        
        else: # Evaluate both C and Y

            #Forward pass
            if cfg.original_model: #The original model returns a list where the first element is the y_hat
                C_hat = XtoC_model(X)

                Y_hat = torch.softmax(CtoY_model(C_hat),dim=1)

            else:

                C_hat = torch.sigmoid(XtoC_model(X))

                Y_hat = torch.softmax(CtoY_model(C_hat),dim=1)

            if C_hat.shape != C_Majority.shape:
                #Apply the mask if the concept is not filtered
                C_hat = C_hat[..., mask]


            #Update concept accuracy
            eval_logger.update_concept_accuracy(mode="Majority", predictions=C_hat, ground_truth=C_Majority)
            eval_logger.update_concept_accuracy(mode="NoMajority", predictions=C_hat, ground_truth=C_noMajority)
            #eval_logger.update_per_concept_accuracy(mode="Majority", predictions=C_hat, ground_truth=C_Majority)
            #eval_logger.update_per_concept_accuracy(mode="NoMajority", predictions=C_hat, ground_truth=C_noMajority)
            

        #Update the class logger
        eval_logger.update_class_accuracy(mode="test",logits=Y_hat, correct_label=Y)
    
    #Calulate sailency score
    if cfg.sailency == True and cfg.mode != "Standard":


        for i in tqdm.tqdm(range(len(Non_majority_data_set))):

            X, C, Y , coordinates = Non_majority_data_set[i]

            X = X.unsqueeze(0)

            
            C = C[mask] #Apply the mask to non majority voted concepts

            coordinates = [coordinates[i] for i in mask]

            concept_list,coordinates = get_visible_consepts(coordinates)

            sailency_maps = get_saliency_maps(X,concept_list,model=XtoC_model,method_type=cfg.sailency_methode)

            sailency_score = saliency_score_image(sailency_maps,coordinates=coordinates)

        #Save the metrics to 
        eval_logger.validate(dir=cfg.output_dir,sailency_score=sailency_score)
    
    else:
        #Save the metrics to 
        eval_logger.validate(dir=cfg.output_dir)
    
    eval_logger.finish()

        
if __name__ == '__main__':
    cofig_dict = OmegaConf.load('config/evaluation.yaml')
    main(cofig_dict)