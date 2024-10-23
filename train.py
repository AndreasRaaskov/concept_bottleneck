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
from analysis import Logger,TrainingLogger



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

    trakker = TrainingLogger(os.path.join(args.log_dir, 'XtoCtrain_log.json'))


    #define the data loaders
    train_transform = get_inception_transform(mode="train",methode="original")
    val_transform = get_inception_transform(mode="val",methode="original")
    
    if args.ckpt:
        #train checkpointed model
        train_data = CUB_dataset(mode='cktp',config_dict=args.CUB_dataloader, transform=train_transform)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = None

    else:
        train_data = CUB_dataset(mode='train',config_dict=args.CUB_dataloader, transform=train_transform)
        val_data = CUB_dataset(mode='val',config_dict=args.CUB_dataloader, transform=val_transform)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    n_concepts = train_data.N_CONCEPTS

    #Write the number of concepts to the logger
    logging.info(f"Number of concepts: {n_concepts}\n")

    model = ModelXtoC(pretrained=args.pretrained, freeze=args.freeze,n_concepts=n_concepts,use_aux=args.use_aux)

    model = model.to(device)

    #Define the loss function
    c_criterion = [] #separate criterion (loss function) for each attribute

    if args.weighted_loss:
        imbalance = train_data.calculate_imbalance()
        for ratio in imbalance:
            # Note this was originally BCEwithLogitsLoss, but I change the output to sigmoid for consistency over all models.
            c_criterion.append(torch.nn.BCELoss(weight=torch.FloatTensor([ratio])).to(device)) 
    else:
        for i in range(n_concepts):
            c_criterion.append(torch.nn.CrossEntropyLoss().to(device))
        
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
                    outputs, aux_outputs = model(X)
                    losses = []

                    for i in range(len(c_criterion)): # Loop over each concept and give an individual loss for each. 
                        losses.append(1.0 * c_criterion[i](outputs[:,i], C[:,i]) + 0.4 * c_criterion[i](aux_outputs[:,i], C[:,i]))
                else: #testing or no aux logits
                    outputs = model(X)
                    losses = []

                    for i in range(len(c_criterion)):
                        losses.append(1.0 * c_criterion[i](outputs[:,i], C[:,i]))
                
                #Calculate accuracy
                trakker.update_concept_accuracy("train",outputs, C)
                trakker.update_loss("train",sum(losses)/n_concepts)

                #Backward pass
                optimizer.zero_grad()
                sum(losses).backward()
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
                        outputs = model(X)

                        #Calculate loss
                        losses = []

                        for i in range(len(c_criterion)):
                            losses.append(1.0 * c_criterion[i](outputs[:,i], C[:,i]))
                        
                        #Calculate accuracy
                        trakker.update_concept_accuracy("val",outputs, C)
                        trakker.update_loss("val",sum(losses)/n_concepts)


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
    trakker = TrainingLogger(os.path.join(args.log_dir, 'C_TO_Y_log.json'))

    device = torch.device(args.device)

    #define the data loaders
    if args.ckpt:
        #train checkpointed model
        train_data = CUB_CtoY_dataset(mode='cktp',config_dict=args.CUB_dataloader, model=XtoC_model) #If XtoC model is provided, use it to generate the concepts
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = None
    else:
        train_data = CUB_CtoY_dataset(mode='train',config_dict=args.CUB_dataloader, model=XtoC_model)
        val_data = CUB_CtoY_dataset(mode='val',config_dict=args.CUB_dataloader)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    num_classes = train_data.N_CLASSES
    num_concepts = train_data.N_CONCEPTS

    #Write the number of classes and concepts to the logger
    logging.info(f"Number of classes: {num_classes}\n")
    logging.info(f"Number of concepts: {num_concepts}\n")

    #Define the model
    model = ModelCtoY(input_dim=train_data.N_CONCEPTS,
                            num_classes=train_data.N_CLASSES)
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
            trakker.update_class_accuracy("train",Yhat, Y)
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
                    trakker.update_class_accuracy("val",Yhat, Y)
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
    trakker = TrainingLogger(os.path.join(args.log_dir, 'train_log.json'))

    device = torch.device(args.device)

        #define the data loaders
    train_transform = get_inception_transform(mode="train",methode="original")
    val_transform = get_inception_transform(mode="val",methode="original")
    
    if args.ckpt:
        #train checkpointed model
        train_data = CUB_dataset(mode='cktp',config_dict=args.CUB_dataloader, transform=train_transform)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = None

    else:
        train_data = CUB_dataset(mode='train',config_dict=args.CUB_dataloader, transform=train_transform)
        val_data = CUB_dataset(mode='val',config_dict=args.CUB_dataloader, transform=val_transform)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    n_concepts = train_data.N_CONCEPTS
    n_classes = train_data.N_CLASSES

    model = ModelXtoCtoY(pretrained=args.pretrained, freeze=args.freeze,
                         n_classes=n_classes, use_aux=args.use_aux, n_concepts=n_concepts)
    model = model.to(device)

    #Define the loss function
    y_criterion = torch.nn.CrossEntropyLoss()

    c_criterion = [] #separate criterion (loss function) for each attribute

    if args.weighted_loss:
        imbalance = train_data.calculate_imbalance()
        for ratio in imbalance:
            # Note this was originally BCEwithLogitsLoss, but I change the output to sigmoid for consistency over all models.
            c_criterion.append(torch.nn.BCELoss(weight=torch.FloatTensor([ratio])).to(device)) 
    else:
        for i in range(n_concepts):
            c_criterion.append(torch.nn.CrossEntropyLoss().to(device))


    #Define the optimizer
    optimizer, scheduler = get_optimizer(model, args)
    stop_epoch = int(math.log(args.min_lr / args.lr) / math.log(args.lr_decay_size)) * args.scheduler_step


    best_val_loss = float('inf')

    #Train the model
    for epoch in range(args.epochs):

        trakker.reset()
        model.train()

        for _, data in enumerate(train_loader):
            
            X,C, Y = data
            X = X.to(device)
            C = C.to(device)
            Y = Y.to(device)

            #Forward pass
            Chat,Yhat, aux_Chat,aux_Yhat = model(X)

            losses = []

            #Calculate y loss
            loss_main = 1.0 * y_criterion(Yhat, Y) + 0.4 * y_criterion(aux_Yhat, Y)
            losses.append(loss_main)

            #Calculate the atribute loss by looping over each prediction. 
            for i in range(len(c_criterion)):
                losses.append(args.attr_loss_weight * (1.0 * c_criterion[i](Chat[:, i], C[:, i]) + 0.4 * c_criterion[i](aux_Chat[:, i], C[:, i])))

            
            loss = sum(losses)

            #Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Calculate accuracy
            trakker.update_class_accuracy("train",Yhat, Y)
            trakker.update_concept_accuracy("train",Chat, C)
            trakker.update_loss("train",loss.item())

        


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
                    loss_main = y_criterion(Yhat, Y)
                    losses.append(loss_main)

                    for i in range(len(c_criterion)):
                        losses.append(args.attr_loss_weight * c_criterion[i](Chat[:, i], C[:, i]))
                    
                    #Calculate concept prediction accuracy
                    trakker.update_concept_accuracy("val",Chat, C)
                    trakker.update_class_accuracy("val",Yhat, Y)
                    trakker.update_loss("val",sum(losses).item())

            val_loss = trakker.get_loss_metrics("val")['avg_loss']
            val_acc = trakker.get_class_metrics("val")['top1_accuracy'] #Acuracy of class prediction


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
    train_transform = get_inception_transform(mode="train",methode="original")
    val_transform = get_inception_transform(mode="val",methode="original")
    
    if args.ckpt:
        #train checkpointed model
        train_data = CUB_dataset(mode='cktp',config_dict=args.CUB_dataloader, transform=train_transform)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = None

    else:
        train_data = CUB_dataset(mode='train',config_dict=args.CUB_dataloader, transform=train_transform)
        val_data = CUB_dataset(mode='val',config_dict=args.CUB_dataloader, transform=val_transform)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    n_classes = train_data.N_CLASSES

    #Write the number of classes and concepts to the logger
    logging.info(f"Number of classes: {train_data.N_CLASSES}\n")

    
    #Define the model
    model = ModelXtoY(pretrained=args.pretrained, freeze=args.freeze,n_classes=n_classes,use_aux=args.use_aux)
    model = model.to(device)

    

    #Define the loss function
    y_criterion = torch.nn.CrossEntropyLoss()

    #Define the optimizer
    optimizer, scheduler = get_optimizer(model, args)
    stop_epoch = int(math.log(args.min_lr / args.lr) / math.log(args.lr_decay_size)) * args.scheduler_step


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
                loss = 1.0 * y_criterion(Yhat, Y) + 0.4 * y_criterion(aux_Yhat, Y)
            else:

                Yhat = model(X)

                #Calculate loss
                loss = y_criterion(Yhat, Y)

            #Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Calculate accuracy
            trakker.update_class_accuracy("train",Yhat, Y)
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
                    trakker.update_class_accuracy("val",Yhat, Y)
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

