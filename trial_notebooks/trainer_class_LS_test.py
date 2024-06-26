# a general trainer function for our classification problem
import torch
import os
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils import tensorboard
from utils import metrics, utilities, spline_utils
import matplotlib.pyplot as plt
from layers.lipschitzlinear import LipschitzLinear
from dataloader.Function_1D import Function1D, generate_testing_set, slope_1_ae, slope_1_flat, cosines, threshold
from activations.linearspline import LinearSpline
from architectures.simple_fc import SimpleFC
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """
    Custom dataset class for our problem.
    """
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y

class Trainer_classification:
    """
    Trainer class for your classification problem.
    """
    def __init__(self, model, xdata, ydata, criterion, config, seed, device, print_after_epoch=5):
        self.model = model
        self.x = xdata
        self.y = ydata
        self.config = config
        self.device = device
        self.criterion = criterion 

        # Split dataset into train and validation sets
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(xdata, ydata,
                                                                    test_size=config["training_options"]["validation_split"], 
                                                                    random_state=seed)
        
        # Print number of DPs of class 0 and 1 in train and val set
        ### let's count number of datapoints for class 0 and 1
        print(f"in Train set: # class 0 DPs: {np.count_nonzero(self.y_train==0)} and # class 1 DPs: {np.count_nonzero(self.y_train==1)}")
        print(f"in val set: # class 0 DPs: {np.count_nonzero(self.y_val==0)} and # class 1 DPs: {np.count_nonzero(self.y_val==1)}")
        # Prepare dataloaders 
        self.train_dataloader = DataLoader(CustomDataset(self.x_train, self.y_train), 
                                        batch_size=config["training_options"]["batch_size"], 
                                        shuffle=True)
        self.val_dataloader = DataLoader(CustomDataset(self.x_val, self.y_val), 
                                        batch_size=config["training_options"]["batch_size"], 
                                        shuffle=False)

        # Set up the optimizer 
        self.set_optimization()

        # average train and val epoch loss
        self.avg_train_loss_epoch=[]
        self.avg_val_loss_epoch = []
        self.train_acc_epoch = []
        self.val_acc_epoch = []


        ## for printing the result
        self.print_after=print_after_epoch

        # Stats to save about the models

    ### setting up the optimizer
    def set_optimization(self):
        """ """
        #for i in range(self.nbr_models):
        params_list = [{'params': spline_utils.get_no_spline_coefficients(self.model), \
                        'lr': self.config["optimizer"]["lr_weights"]}]
        if self.model.using_splines:
            params_list.append({'params': spline_utils.get_spline_coefficients(self.model), \
                                'lr': self.config["optimizer"]["lr_spline_coeffs"]})

            if self.config["activation_fn_params"]["spline_scaling_coeff"]:
                params_list.append({'params': spline_utils.get_spline_scaling_coeffs(self.model), \
                                    'lr': self.config["optimizer"]["lr_spline_scaling_coeffs"]})
        self.optimizer = torch.optim.Adam(params_list, weight_decay=0.0001)

    def train(self):
        """
        Main training loop.
        """
        for epoch in range(self.config["training_options"]["epochs"]):
            self.train_epoch(epoch)
            self.validate_epoch(epoch)
            if (epoch+1) % self.print_after ==0 :
                print("_"*30)
                print(f"avg epoch train loss: {self.avg_train_loss_epoch[epoch]:.6f} and validation loss: {self.avg_val_loss_epoch[epoch]:.6f}")
                print(f"training acc: {self.train_acc_epoch[epoch]:.6f} and validation acc: {self.val_acc_epoch[epoch]:.6f}")
                print("_"*30)
        # Need to add Additional post-training actions here
        ### (right now I dont wanna save my checkpoints)

    def train_epoch(self, epoch):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_train_loss=0
        tbar = tqdm(self.train_dataloader)
        for batch_idx, (data, target) in enumerate(tbar): #(self.train_dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            output = self.model(data)### output shape: (N,1)
            output = output.squeeze()

            # Compute loss
            data_loss = self.criterion(output, target.float())

            # TV2 regulatisation
            regularization=0
            if self.model and self.config['activation_fn_params']['lmbda'] > 0:
                regularization = self.config['activation_fn_params']['lmbda'] * self.model.TV2()
            # total loss
            total_loss = data_loss + regularization#data_loss #+ regularization
            ### Yippie I figured out the issue. it is with the data_loss term 

            # Backward pass and optimization
            self.optimizer.zero_grad()
            total_loss.backward()
            # update the parameters
            self.optimizer.step()

            total_train_loss+=total_loss
            # If need to add code to Log training progress
            # my training progress logging code goes here

        # average training loss for the epoch
        avg_train_loss = total_train_loss / len(self.train_dataloader)
        self.avg_train_loss_epoch.append(avg_train_loss.detach())

        # evaluating training accuracy
        with torch.no_grad():
            pred= self.model(torch.tensor(self.x_train))
            acc_train  = (pred.squeeze().round() == torch.tensor(self.y_train) ).float().mean()
            self.train_acc_epoch.append(acc_train)
        

    def validate_epoch(self, epoch):
        """
        Validate the model for one epoch.
        """
        self.model.eval()
        total_val_loss=0
        # Validation loop
        with torch.no_grad():
            for data, target in self.val_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                output = output.squeeze()

                # Compute validation metrics (e.g., accuracy, loss)
                data_loss = self.criterion(output, target.float())
                total_val_loss+=data_loss

            avg_val_loss = total_val_loss / len(self.val_dataloader)
            self.avg_val_loss_epoch.append(avg_val_loss.detach())
            # # Calculating accuracy of model
            # acc_val  = (output.round() == target).float().mean
            # self.val_acc_epoch.append(acc_val)
            # evaluating training accuracy
            pred= self.model(torch.tensor(self.x_val))
            acc_val  = (pred.squeeze().round() == torch.tensor(self.y_val) ).float().mean()
            self.val_acc_epoch.append(acc_val)
            # Additional validation actions go here
        

    def save_checkpoint(self, epoch):
        """
        Save model checkpoint.
        """
        # Your checkpoint saving code goes here

    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint.
        """
        # Your checkpoint loading code goes here

    # Additional methods for logging, evaluation, etc. go here
