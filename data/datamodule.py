from torch.utils.data import DataLoader
from torchvision import transforms

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
        sorted_dataset=False,
    ):
        
        print ("coucou 1")
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
            transforms=test_transform, #pas de data augmentation
            sorted=sorted_dataset,
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
           
        self.train_set = Dataset(
            self.dataset_path,
            "train_val",
            transforms=self.train_transform,
            indices=self.train_idx,
        )
        self.val_set = Dataset(
            self.dataset_path,
            "train_val",
            transforms=self.test_transform,
            indices=self.val_idx,
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
        """Validation dataloader."""
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
    
    def train_high_dataloader(self, threshold=12):
        """Train dataloader with high targets only."""
        # Récupérer les cibles (targets) du train_set
        targets = np.array([self.full_dataset.targets[i] for i in self.train_set.indices])
        idx_high = np.where(targets > threshold)[0]

        high_set = Subset(self.train_set, idx_high)

        return DataLoader(
            high_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
    def train_low_dataloader(self, threshold=12):
        """Train dataloader with low targets only."""
        # Récupérer les cibles (targets) du train_set
        targets = np.array([self.full_dataset.targets[i] for i in self.train_set.indices])
        idx_low = np.where(targets < threshold)[0]

        low_set = Subset(self.train_set, idx_low)

        return DataLoader(
            low_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
    

class DataModuleTemporal(DataModule):
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
        sliding_window=False,
    ):
        if train_idx is not None or val_idx is not None:
            raise ValueError("train_idx and val_idx should not be provided for DataModuleTemporal.")
        self.sliding_window = sliding_window
        self.full_dataset = Dataset(
            dataset_path,
            "train_val",
            transforms=train_transform, #pas de data augmentation
            sorted=sliding_window,
        )
        print("Entrainement temporalisé")
        idx_2023 = np.nonzero(np.array(self.full_dataset.year) == 2023)[0]
        idx_2022 = np.nonzero(np.array(self.full_dataset.year) == 2022)[0]
        selected_2022 = np.random.choice(idx_2022, size=len(idx_2022)//2, replace=False,)
        # Ajouter à val_idx
        # Keyvan, je trouve pas ca cohérent de prendre un validation set aussi spécifique
        self.val_idx = np.concatenate([selected_2022, idx_2023])
        self.train_idx = np.setdiff1d(np.arange(len(self.full_dataset)), self.val_idx)
        

        val_percentage = len(self.val_idx) / len(self.full_dataset) * 100
        print(f"Pourcentage des données utilisées pour la validation : {val_percentage:.2f}%")

        self.anneeMin = -1

        super().__init__(
            dataset_path,
            train_transform,
            test_transform,
            batch_size,
            num_workers,
            taille_val,
            train_idx=self.train_idx,
            val_idx=self.val_idx,
            sorted_dataset=sliding_window,
        )

    def train_val_dataloader(self):
        return super().train_val_dataloader()
    
    def val_dataloader(self):
        return super().val_dataloader()
    
    def test_dataloader(self):
        return super().test_dataloader()
    
    def train_dataloader(self):
        """Train dataloader."""
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=not self.sliding_window,
            num_workers=self.num_workers,
        )
    
    
    
    def train_dataloader_dynamique(self, epoch = 0):
        """Train dataloader dynamique. Change la fenêtre sur chaque epoch"""
        
        if (epoch // 5 < (2023 - 2011)):
            anneeMin = 2011 + epoch // 5
        else:
            anneeMin = 2022
        
        
        if (epoch != 0 and (self.anneeMin == -1 or self.anneeMin != anneeMin)) :
            print ("******")
            print ("DATAMODULE DYNAMIQUE")
            print ("******")

            print ("Nouvelle année minimum : ", anneeMin)
        
            new_train_idx = []

            for idx in self.train_idx:
                if self.full_dataset[idx]["year"] >= anneeMin:
                    new_train_idx.append(idx)
            
            self.train_idx = new_train_idx

        self.anneeMin = anneeMin

        return (self.train_dataloader ())
        
class DataModuleModern(DataModule):
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
        sliding_window=False
    ):
        
        if train_idx is not None or val_idx is not None:
            raise ValueError("train_idx and val_idx should not be provided for DataModuleModern.")
        
        print("Entrainement moderne")

        self.sliding_window = False
        self.full_dataset = Dataset(
            dataset_path,
            "train_val",
            transforms=train_transform,
            sorted=False,
        )
        idx_2023 = np.nonzero(np.array(self.full_dataset.year) == 2023)[0]
        idx_2022 = np.nonzero(np.array(self.full_dataset.year) == 2022)[0]

        indices = np.concatenate([idx_2022, idx_2023])
        years = np.array(self.full_dataset.year)[indices]
        val_size = int(taille_val * len(indices))

        self.train_idx, self.val_idx = train_test_split(
                indices,
                test_size=val_size,
                stratify=years
            )

        super().__init__(
            dataset_path,
            train_transform,
            test_transform,
            batch_size,
            num_workers,
            taille_val,
            train_idx=self.train_idx,
            val_idx=self.val_idx,
            sorted_dataset=False,
        )

    def val_dataloader(self):
        return super().val_dataloader()
    
    def test_dataloader(self):
        return super().test_dataloader()
    
    def train_dataloader(self):
        """Train dataloader."""
        return super().train_dataloader()

