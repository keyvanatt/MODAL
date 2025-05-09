from torch.utils.data import DataLoader

from data.dataset import Dataset

from torch.utils.data import random_split

class DataModule:
    def __init__(
        self,
        dataset_path,
        train_transform,
        test_transform,
        batch_size,
        num_workers,
        taille_val,
        metadata=["views"],
    ):
        self.dataset_path = dataset_path
        self.train_transform = train_transform  
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metadata = metadata
        self.taille_val = taille_val
        
        full_dataset = Dataset(
            self.dataset_path,
            "train_val",
            transforms=self.test_transform,  
            metadata=self.metadata,
        )

        # Le self.taillee_val est accessible dans le train.yaml. A modifier en fonction des besoins/envies
        val_size = int(self.taille_val * len(full_dataset))
        train_size = len(full_dataset) - val_size
        # random split va couper de façon aléatoire le dataset en sous-dataset disjoints 
        # de taille train_size puis val_size. L'idée c'est de prendre aléatoirement des éléments
        # du train data pour en faire dataset de validation.
        # On n'a pas besoin de récupérer le "dataset d'entrainement" généré par rendom split d'où le _ 
        self.train_set, self.val_set = random_split(full_dataset, [train_size, val_size])
       
        

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
            metadata=self.metadata,
        )
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )