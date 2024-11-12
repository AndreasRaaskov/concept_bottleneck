import pdb
import os
import sys

import math
import torch
from torch.utils.data import DataLoader
import numpy as np
import logging


#from dataset import load_data, find_class_imbalance
from data_loaders import CUB_dataset,CUB_CtoY_dataset
from models import   ModelXtoY, ModelXtoC, ModelXtoCtoY, ModelCtoY,get_inception_transform
from utils.analysis import TrainingLogger



def get_optimizer(model, args):
    """
    Define the optimizer and scheduler based on the arguments
    """
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.lr_decay_size)
    return optimizer, scheduler



def train_X_to_C(args):
    """
    Train concept prediction model used in independent and sequential training
    """

    device = torch.device(args.device)

    #Define the loggers
    #logger = Logger(os.path.join(args.log_dir, 'log.txt'))
    #logger.write(str(args) + '\n')

    trakker = TrainingLogger(os.path.join(args.log_dir, 'XtoC_log.json'))


    #define the data loaders
    train_transform = get_inception_transform(mode="train",methode=args.transform_method)
    val_transform = get_inception_transform(mode="val",methode=args.transform_method)
    
    if args.ckpt:
        #train checkpointed model
        train_data = CUB_dataset(mode='ckpt',config_dict=args.CUB_dataloader, transform=train_transform)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = None

    else:
        train_data = CUB_dataset(mode='train',config_dict=args.CUB_dataloader, transform=train_transform)
        val_data = CUB_dataset(mode='val',config_dict=args.CUB_dataloader, transform=val_transform)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    n_concepts = train_data.n_concepts

    #Write the number of concepts to the logger
    logging.info(f"Number of concepts: {n_concepts}\n")

    model = ModelXtoC(pretrained=args.pretrained, freeze=args.freeze,n_concepts=n_concepts,use_aux=args.use_aux)

    model = model.to(device)

    #Define the loss function
    if args.weighted_loss:
        imbalance = train_data.calculate_imbalance()
        pos_weights = torch.tensor(imbalance).to(device)
        c_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights,reduction='sum')
    else:
        c_criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
            
    #Define the optimizer
    optimizer, scheduler = get_optimizer(model, args)

    best_val_loss = float('inf')
    best_val_epoch = -1

    #Train the model
    for epoch in range(args.epochs):
            
            trakker.reset()
            model.train()
    
            for _, data in enumerate(train_loader):
                X, C, _ = data

                C = C.to(device)
                X = X.to(device)

                #Calculate loss
                if args.use_aux:
                    Chat, aux_Chat = model(X)
                    
                    main_loss = c_criterion(Chat, C) 
                    aux_loss = c_criterion(aux_Chat, C)

                    loss = main_loss + 0.4 * aux_loss
                    trakker.update_loss("train",main_loss)

                else: #testing or no aux logits
                    outputs = model(X)

                    loss = c_criterion(Chat, C)
                    trakker.update_loss("train",loss)
                
                #Calculate accuracy
                trakker.update_concept_accuracy("train",torch.sigmoid(Chat), C)
                

                #Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            

            #Evaluate the model
            if not args.ckpt:
                model.eval()
                with torch.no_grad():
                    for _, data in enumerate(val_loader):
                        X, C, _ = data
                        C = C.to(device)
                        X = X.to(device)

                        #Forward pass
                        Chat = model(X)

                        #Calculate loss
                        loss = c_criterion(Chat, C)


                        #Calculate accuracy
                        trakker.update_concept_accuracy("val",torch.sigmoid(Chat), C)
                        trakker.update_loss("val",loss)


                    #Check if the model is the best model
                    val_loss = trakker.get_loss_metrics("val")['avg_loss']
                    val_acc = trakker.get_concept_metrics("val")['accuracy']

                    logging.info(f"Epoch [{epoch:2d}]: val loss: {val_loss:.4f} val acc: {val_acc:.4f}\n")


            else:
                #If the model is a checkpointed model, only evaluate the model on the training set
                val_loss = trakker.get_loss_metrics("train")['avg_loss']
                val_acc = trakker.get_concept_metrics("train")['accuracy']
                
                logging.info(f"Epoch [{epoch:2d}]: train loss: {val_loss:.4f} train acc: {val_acc:.4f}\n")

            #Save all the metrics to json file
            trakker.log_metrics(epoch)
            
            if val_loss < best_val_loss: # Note: the original code used accuracy as the metric for early stopping
                    logging.info(f"New best model at epoch {epoch}\n")
                    best_val_epoch = epoch
                    best_val_loss = val_loss
                    torch.save(model, os.path.join(args.log_dir,'best_XtoC_model.pth'))


            # Update the learning rate
            scheduler.step()
            
            # Check if we've reached the minimum learning rate
            if optimizer.param_groups[0]['lr'] <= args.min_lr:
                optimizer.param_groups[0]['lr'] = args.min_lr
            
            # Log the learning rate every 10 epochs
            if epoch % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch}, Current lr: {current_lr}')

            if epoch - best_val_epoch >= 100:
                logging.info("Early stopping because acc hasn't improved for a long time")
                break
    
    #Return the best model
    return torch.load(os.path.join(args.log_dir,'best_XtoC_model.pth'))
            


                    
    


def train_C_to_Y(args,XtoC_model=None):
    """
    train the C to Y model used in independent and sequential training
    if a CtoY model is provided, the model is trained using the provided model to generate the concepts (sequential training) else use concept given by data (independent training)
    """

    #Define the loggers
    trakker = TrainingLogger(os.path.join(args.log_dir, 'CtoY_log.json'))

    device = torch.device(args.device)

    #Get the validation tranformation
    transform = get_inception_transform(mode="val",methode=args.transform_method)

    #define the data loaders
    if args.ckpt:
        #train checkpointed model
        train_data = CUB_CtoY_dataset(mode='ckpt',config_dict=args.CUB_dataloader,transform=transform, model=XtoC_model) #If XtoC model is provided, use it to generate the concepts
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = None
    else:
        train_data = CUB_CtoY_dataset(mode='train',config_dict=args.CUB_dataloader,transform=transform, model=XtoC_model)
        val_data = CUB_CtoY_dataset(mode='val',config_dict=args.CUB_dataloader)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    num_classes = train_data.n_classes
    num_concepts = train_data.n_concepts

    #Write the number of classes and concepts to the logger
    logging.info(f"Number of classes: {num_classes}\n")
    logging.info(f"Number of concepts: {num_concepts}\n")

    #Define the model
    model = ModelCtoY(input_dim=num_concepts,
                            num_classes=num_classes)
    model = model.to(device)

    

    #Define the loss function
    y_criterion = torch.nn.CrossEntropyLoss()

    #Define the optimizer
    optimizer, scheduler = get_optimizer(model, args)

    best_val_loss = float('inf')

    #Train the model
    for epoch in range(args.epochs):

        trakker.reset()
        model.train()

        for _, data in enumerate(train_loader):
            
            C, Y = data
            C = C.to(device)
            Y = Y.to(device)

            #Forward pass
            Yhat = model(C)

            #Calculate loss
            loss = y_criterion(Yhat, Y)

            #Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Calculate accuracy
            trakker.update_class_accuracy("train",torch.softmax(Yhat,axis=1), Y)
            trakker.update_loss("train",loss)
        


        #Evaluate the model
        if not args.ckpt:
            model.eval()
            with torch.no_grad():
                for _, data in enumerate(val_loader):
                    C, Y = data
                    C = C.to(device)
                    Y = Y.to(device)

                    #Forward pass
                    Yhat = model(C)

                    #Calculate loss
                    loss = y_criterion(Yhat, Y)

                    #Calculate accuracy
                    trakker.update_class_accuracy("val",torch.softmax(Yhat,axis=1), Y)
                    trakker.update_loss("val",loss)
                
                #Check if the model is the best model
                val_loss = trakker.get_loss_metrics("val")['avg_loss']
                val_acc = trakker.get_class_metrics("val")['top1_accuracy']

                logging.info(f"Epoch [{epoch:2d}]: val loss: {val_loss:.4f} val acc: {val_acc:.4f}\n")
        else:
            #If the model is a checkpointed model, only evaluate the model on the training set
            val_loss = trakker.get_loss_metrics("train")['avg_loss']
            val_acc = trakker.get_class_metrics("train")['top1_accuracy']
            
            logging.info(f"Epoch [{epoch:2d}]: train loss: {val_loss:.4f} train acc: {val_acc:.4f}\n")

        #Save all the metrics to json file
        trakker.log_metrics(epoch)
        
        if val_loss < best_val_loss: # Note: the original code used accuracy as the metric for early stopping
                logging.info(f"New best model at epoch {epoch}\n")
                best_val_epoch = epoch
                best_val_loss = val_loss
                torch.save(model, os.path.join(args.log_dir,'best_CtoY_model.pth'))


        # Update the learning rate
        scheduler.step()
        
        # Check if we've reached the minimum learning rate
        if optimizer.param_groups[0]['lr'] <= args.min_lr:
            optimizer.param_groups[0]['lr'] = args.min_lr
        
        # Log the learning rate every 10 epochs
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch}, Current lr: {current_lr}')

        if epoch - best_val_epoch >= 100:
            logging.info("Early stopping because acc hasn't improved for a long time")
            break
        



def train_X_to_C_to_y(args):
    """
    Joint training
    """

    #Define the loggers
    XtoC_trakker = TrainingLogger(os.path.join(args.log_dir, 'XtoC_log.json'))
    CtoY_trakker = TrainingLogger(os.path.join(args.log_dir, 'CtoY_log.json'))

    device = torch.device(args.device)

        #define the data loaders
    train_transform = get_inception_transform(mode="train",methode=args.transform_method)
    val_transform = get_inception_transform(mode="val",methode=args.transform_method)
    
    if args.ckpt:
        #train checkpointed model
        train_data = CUB_dataset(mode='ckpt',config_dict=args.CUB_dataloader, transform=train_transform)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = None

    else:
        train_data = CUB_dataset(mode='train',config_dict=args.CUB_dataloader, transform=train_transform)
        val_data = CUB_dataset(mode='val',config_dict=args.CUB_dataloader, transform=val_transform)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    n_concepts = train_data.n_concepts
    n_classes = train_data.n_classes

    model = ModelXtoCtoY(pretrained=args.pretrained, freeze=args.freeze,
                         n_classes=n_classes, use_aux=args.use_aux, n_concepts=n_concepts)
    model = model.to(device)

    #Define the loss function
    y_criterion = torch.nn.CrossEntropyLoss()

    #Define the loss function
    if args.weighted_loss:
        imbalance = train_data.calculate_imbalance()
        # Create pos_weights tensor for all concepts at once [num_concepts]
        pos_weights = torch.tensor(imbalance).to(device)
        c_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights,reduction='sum')
    else:
        c_criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')


    #Define the optimizer
    optimizer, scheduler = get_optimizer(model, args)


    best_val_acc = 0

    #Train the model
    for epoch in range(args.epochs):

        XtoC_trakker.reset()
        CtoY_trakker.reset()

        model.train()

        for _, data in enumerate(train_loader):
            
            X,C, Y = data
            X = X.to(device)
            C = C.to(device)
            Y = Y.to(device)

            #Forward pass
            if args.use_aux:
                Chat,Yhat, aux_Chat,aux_Yhat = model(X)

                #Calculate y loss
                class_loss = y_criterion(Yhat, Y)
                class_aux_loss = y_criterion(aux_Yhat, Y)

                #Calculate the atribute loss by looping over each prediction, and multiply by lambda 
                concept_loss = c_criterion(Chat,C) 
                concept_aux_loss = c_criterion(aux_Chat,C) 

                #Calculate the main loss as y loss * lambda *sum (concept losses))
                main_loss = class_loss + concept_loss* args.lambda1
                aux_loss = class_aux_loss + concept_aux_loss* args.lambda1

                loss = main_loss + 0.4 * aux_loss
                CtoY_trakker.update_loss("train",class_loss)
                XtoC_trakker.update_loss("train",concept_loss)
            else:
                Chat,Yhat = model(X)

                #Calculate y loss
                main_loss = y_criterion(Yhat, Y)

                #Calculate the atribute loss by looping over each prediction, and multiply by lambda
                concept_loss = c_criterion(Chat,C)

                loss = main_loss + concept_loss* args.lambda1

                CtoY_trakker.update_loss("train",class_loss)
                XtoC_trakker.update_loss("train",concept_loss)

            #Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Calculate accuracy
            CtoY_trakker.update_class_accuracy("train",torch.softmax(Yhat,axis=1), Y)
            XtoC_trakker.update_concept_accuracy("train",torch.sigmoid(Chat), C)
            
        #Evaluate the model
        if not args.ckpt:
            model.eval()
            with torch.no_grad():
                for _, data in enumerate(val_loader):
                    X,C, Y = data
                    C = C.to(device)
                    X = X.to(device)
                    Y = Y.to(device)

                    #Forward pass
                    Chat,Yhat = model(X)


                    #Calculate y loss
                    class_loss = y_criterion(Yhat, Y)

                    #Caluate c loss
                    concept_loss = c_criterion(Chat,C)

                    loss = class_loss + concept_loss* args.lambda1
                    
                    #Calculate concept prediction accuracy
                    XtoC_trakker.update_concept_accuracy("val",torch.sigmoid(Chat), C)
                    XtoC_trakker.update_loss("val",concept_loss)

                    CtoY_trakker.update_class_accuracy("val",torch.softmax(Yhat,axis=1), Y)
                    CtoY_trakker.update_loss("val",class_loss)

            val_loss = CtoY_trakker.get_loss_metrics("val")['avg_loss']
            val_acc = CtoY_trakker.get_class_metrics("val")['top1_accuracy'] #Acuracy of class prediction


        else:
            #If the model is a checkpointed model, only evaluate the model on the training set
            val_loss = CtoY_trakker.get_loss_metrics("train")['avg_loss']
            val_acc = CtoY_trakker.get_class_metrics("train")['top1_accuracy']
            
            logging.info(f"Epoch [{epoch:2d}]: train loss: {val_loss:.4f} train acc: {val_acc:.4f}\n")

        #Save all the metrics to json file
        XtoC_trakker.log_metrics(epoch)
        CtoY_trakker.log_metrics(epoch)
        
        if val_acc > best_val_acc: # Note: the original code used accuracy as the metric for early stopping
                logging.info(f"New best model at epoch {epoch}\n")
                best_val_epoch = epoch
                best_val_acc = val_acc
                torch.save(model, os.path.join(args.log_dir, 'best_Joint_model.pth'))


        # Update the learning rate
        scheduler.step()
        
        # Check if we've reached the minimum learning rate
        if optimizer.param_groups[0]['lr'] <= args.min_lr:
            optimizer.param_groups[0]['lr'] = args.min_lr
        
        # Log the learning rate every 10 epochs
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch}, Current lr: {current_lr}')

        if epoch - best_val_epoch >= 100:
            logging.info("Early stopping because acc hasn't improved for a long time")
            break

def train_X_to_y(args):
    """
    A standard model that predicts class labels using only images with CNN
    """

    #Define the loggers
    trakker = TrainingLogger(os.path.join(args.log_dir, 'train_log.json'))

    device = torch.device(args.device)

    #define the data loaders
    train_transform = get_inception_transform(mode="train",methode=args.transform_method)
    val_transform = get_inception_transform(mode="val",methode=args.transform_method)
    
    if args.ckpt:
        #train checkpointed model
        train_data = CUB_dataset(mode='ckpt',config_dict=args.CUB_dataloader, transform=train_transform)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = None

    else:
        train_data = CUB_dataset(mode='train',config_dict=args.CUB_dataloader, transform=train_transform)
        val_data = CUB_dataset(mode='val',config_dict=args.CUB_dataloader, transform=val_transform)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    n_classes = train_data.n_classes

    #Write the number of classes and concepts to the logger
    logging.info(f"Number of classes: {n_classes}\n")

    
    #Define the model
    model = ModelXtoY(pretrained=args.pretrained, freeze=args.freeze,n_classes=n_classes,use_aux=args.use_aux)
    model = model.to(device)

    

    #Define the loss function
    y_criterion = torch.nn.CrossEntropyLoss()

    #Define the optimizer
    optimizer, scheduler = get_optimizer(model, args)


    best_val_loss = float('inf')

    #Train the model
    for epoch in range(args.epochs):

        trakker.reset()
        model.train()

        for _, data in enumerate(train_loader):
            
            X,_, Y = data
            X = X.to(device)
            Y = Y.to(device)

            #Forward pass
            if args.use_aux:
                Yhat, aux_Yhat = model(X)

                #Calculate y loss
                main_loss = y_criterion(Yhat, Y)
                aux_loss = y_criterion(aux_Yhat, Y)
                loss = main_loss + 0.4 * aux_loss
                trakker.update_loss("train",main_loss) #only log the main loss
            else:
                Yhat = model(X)

                #Calculate loss
                loss = y_criterion(Yhat, Y)
                trakker.update_loss("train",loss)

            #Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Calculate accuracy
            trakker.update_class_accuracy("train",torch.softmax(Yhat,dim=1), Y)
        

        #Evaluate the model
        if not args.ckpt:
            model.eval()
            with torch.no_grad():
                for _, data in enumerate(val_loader):
                    X,_, Y = data
                    X = X.to(device)
                    Y = Y.to(device)

                    #Forward pass
                    Yhat = model(X)

                    #Calculate loss
                    loss = y_criterion(Yhat, Y)

                    #Calculate accuracy
                    trakker.update_class_accuracy("val",torch.softmax(Yhat,dim=1), Y)
                    trakker.update_loss("val",loss)
                
                #Check if the model is the best model
                val_loss = trakker.get_loss_metrics("val")['avg_loss']
                val_acc = trakker.get_class_metrics("val")['top1_accuracy']

                logging.info(f"Epoch [{epoch:2d}]: val loss: {val_loss:.4f} val acc: {val_acc:.4f}\n")
        else:
            #If the model is a checkpointed model, only evaluate the model on the training set
            val_loss = trakker.get_loss_metrics("train")['avg_loss']
            val_acc = trakker.get_class_metrics("train")['top1_accuracy']
            
            logging.info(f"Epoch [{epoch:2d}]: train loss: {val_loss:.4f} train acc: {val_acc:.4f}\n")

        #Save all the metrics to json file
        trakker.log_metrics(epoch)
        
        if val_loss < best_val_loss: # Note: the original code used accuracy as the metric for early stopping
                logging.info(f"New best model at epoch {epoch}\n")
                best_val_epoch = epoch
                best_val_loss = val_loss
                torch.save(model, os.path.join(args.log_dir,'best_XtoY_model.pth'))


        # Update the learning rate
        scheduler.step()
        
        
        # Log the learning rate every 10 epochs
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch}, Current lr: {current_lr}')

        if epoch - best_val_epoch >= 100:
            logging.info("Early stopping because acc hasn't improved for a long time")
            break

