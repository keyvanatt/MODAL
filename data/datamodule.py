from torch.utils.data import DataLoader

from data.dataset import Dataset

from torch.utils.data import random_split,Subset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


class DataModule:
    def __init__(
        self,
        dataset_path,
        train_transform,
        test_transform,
        batch_size,
        num_workers,
        taille_val,
        train_idx=None,
        val_idx=None,
    ):
        self.dataset_path = dataset_path
        self.train_transform = train_transform  
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.taille_val = taille_val
        self.train_idx = train_idx
        self.val_idx = val_idx
        
        self.full_dataset = Dataset(
            self.dataset_path,
            "train_val",
            transforms=self.test_transform,  
        )

        if self.train_idx is not None and self.val_idx is not None:
            print("Utilisation des indices fournis pour créer les ensembles d'entraînement et de validation.")
            
        else:

            # Le self.taillee_val est accessible dans le train.yaml. A modifier en fonction des besoins/envies
            val_size = int(self.taille_val * len(self.full_dataset))
            train_size = len(self.full_dataset) - val_size
            # random split va couper de façon aléatoire le dataset en sous-dataset disjoints 
            # de taille train_size puis val_size. L'idée c'est de prendre aléatoirement des éléments
            # du train data pour en faire dataset de validation.
            # On n'a pas besoin de récupérer le "dataset d'entrainement" généré par rendom split d'où le _ 
            # Créer un tableau d'indices de bins pour chaque target
            indices = np.arange(len(self.full_dataset))
            # Stratify by both years and YouTube channels
            years = np.array(self.full_dataset.year) 
            channels = np.array(self.full_dataset.channel) 

            self.train_idx, self.val_idx = train_test_split(
                indices,
                test_size=val_size,
                stratify=years
            )
           
            
        self.train_set = Subset(self.full_dataset, self.train_idx)
        self.val_set = Subset(self.full_dataset, self.val_idx)

        # Plot the distribution of targets in the training set
        train_targets = np.array(self.full_dataset.targets)[self.train_idx]
        plt.figure(figsize=(8, 4))
        plt.hist(train_targets, bins=100, edgecolor='black')
        plt.title("Distribution of Targets in Train Set")
        plt.xlabel("Target")
        plt.ylabel("Count")
        plt.show()

        # Plot the distribution of targets in the validation set
        val_targets = np.array(self.full_dataset.targets)[self.val_idx]
        plt.figure(figsize=(8, 4))
        plt.hist(val_targets, bins=100, edgecolor='black')
        plt.title("Distribution of Targets in Validation Set")
        plt.xlabel("Target")
        plt.ylabel("Count")
        plt.show()

        # Plot the distribution of targets in the training set
        train_targets = np.array(self.full_dataset.targets)[self.train_idx]
        plt.figure(figsize=(8, 4))
        plt.hist(train_targets, bins=len(np.unique(train_targets)), edgecolor='black')
        plt.title("Distribution of Targets in Train Set")
        plt.xlabel("Target")
        plt.ylabel("Count")
        plt.show()

       
    def train_val_dataloader(self):
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        ) 

    def train_dataloader(self):
        """Train dataloader."""
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """
        Ancien TODO. On prend aléatoirement 10% du training set pour en faire un validation set
        que l'on renvoie au format DataLoader de PyTorch.
        
        TODO: 
        Implement a strategy to create a validation set from the train set.
        """
        
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
     
    def test_dataloader(self):
        """Test dataloader."""
        # Hérite de torch.utils.data.Dataset
        dataset = Dataset(
            self.dataset_path,
            "test",
            transforms=self.test_transform,
        )
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )