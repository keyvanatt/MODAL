import torch
import pandas as pd
from PIL import Image
import numpy as np
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transforms, sorted=False, indices=None):
        
        self.dataset_path = dataset_path
        self.split = split
        # - read the info csvs
        print(f"{dataset_path}/{split}.csv")
        info = pd.read_csv(f"{dataset_path}/{split}.csv")
        info["description"] = info["description"].fillna("")

        if indices is not None:
            # - filter the dataset by indices
            info = info.iloc[indices]

        if "views" in info.columns:
            self.targets = info["log1p_views"].values
        if "category" in info.columns:
            self.class_targets = info["category"].values            
        

        # - ids
        self.ids = info["id"].values
        # - text
        self.description = info["description"].values
        self.title = info["title"].values
        
        # tabular data
        self.channel = info["channel_id"].values
        self.year = info["year"].values
        
        self.http_count = info["http_count"].values

        self.diese = info["diese"].values
        self.nb_mots = info["nb_mots"].values

        # - transforms
        self.transforms = transforms

        # sort the dataset by year
        if sorted:
            sorted_indices = self.year.argsort()
            self.ids = self.ids[sorted_indices]
            self.description = self.description[sorted_indices]
            self.title = self.title[sorted_indices]
            self.channel = self.channel[sorted_indices]
            self.year = self.year[sorted_indices]

            if hasattr(self, "targets"):
                self.targets = self.targets[sorted_indices]

            if hasattr(self, "class_targets"):
                self.class_targets = self.class_targets[sorted_indices]

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        # - load the image
        image = Image.open(
            f"{self.dataset_path}/{self.split}/{self.ids[idx]}.jpg"
        ).convert("RGB")
        image = self.transforms(image)
        value = {
            "id": self.ids[idx],
            "image": image,
            "title": self.title[idx],
            "description": self.description[idx],
            "channel": torch.tensor([self.channel[idx]]),
            "year": torch.tensor([self.year[idx]]),
            "http_count": torch.tensor([self.http_count[idx]]),
            "diese": torch.tensor([self.diese[idx]]),
            "nb_mots": torch.tensor([self.nb_mots[idx]])
        }
        # - don't have the target for test
        if hasattr(self, "targets"):
            value["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        if hasattr(self, "class_targets"):
            value["class_target"] = torch.tensor(self.class_targets[idx], dtype=torch.int64)
        return value
    