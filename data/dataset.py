import torch
import pandas as pd
from PIL import Image
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transforms, sorted=False):
        
        self.dataset_path = dataset_path
        self.split = split
        # - read the info csvs
        print(f"{dataset_path}/{split}.csv")
        info = pd.read_csv(f"{dataset_path}/{split}.csv")
        info["description"] = info["description"].fillna("")

        if "views" in info.columns:
            self.targets = info["log1p_views"].values
        
        

        # - ids
        self.ids = info["id"].values
        # - text
        self.description = info["description"].values
        self.title = info["title"].values
        
        # tabular data
        self.channel = info["channel_id"].values
        self.year = info["year"].values
        
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
            "year": torch.tensor([self.year[idx]])
        }
        # - don't have the target for test
        if hasattr(self, "targets"):
            value["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return value
    