import torch
import pandas as pd
from PIL import Image
import numpy as np

class DatasetCLIP(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transforms, sorted=False, indices=None, min_views=None):
        self.dataset_path = dataset_path
        self.split = split

        # Lecture du CSV
        print(f"{dataset_path}/{split}.csv")
        info = pd.read_csv(f"{dataset_path}/{split}.csv")
        info["description"] = info["description"].fillna("")

        
        # Filtrage sur le nombre de vues
        if min_views is not None:
            if "log1p_views" in info.columns:
                info = info[info["log1p_views"] >= min_views]
            if "views" in info.columns:
                info = info[info["views"] >= min_views]
        print(f"Nombre d'exemples après filtrage min_views={min_views} : {len(info)}")

        # Filtrage par indices (après le filtrage min_views)
        if indices is not None:
            info = info.iloc[indices]
            self.indices = indices
        else:
            self.indices = np.arange(info.shape[0])

        # TOUS les attributs sont définis à partir de info filtré
        # Je pète un cable
        if "log1p_views" in info.columns:
            self.targets = info["log1p_views"].values
        elif "views" in info.columns:
            self.targets = np.log1p(info["views"].values)
        if "category" in info.columns:
            self.class_targets = info["category"].values

        # Attributs
        self.ids = info["id"].values
        self.description = info["description"].astype(str).values  # Assure que c'est bien du str
        self.title = info["title"].astype(str).values if "title" in info.columns else None
        self.channel = info["channel_id"].values
        self.year = info["year"].values
        self.http_count = info["http_count"].values if "http_count" in info.columns else np.zeros(len(self.ids))
        self.diese = info["diese"].values if "diese" in info.columns else np.zeros(len(self.ids))
        self.nb_mots = info["nb_mots"].values if "nb_mots" in info.columns else np.zeros(len(self.ids))

        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Chargement et préprocessing de l'image
        image = Image.open(f"{self.dataset_path}/{self.split}/{self.ids[idx]}.jpg").convert("RGB")
        image = self.transforms(image)

        value = {
            "id": self.ids[idx],
            "image": image,
            "description": self.description[idx],
            "channel": torch.tensor([self.channel[idx]], dtype=torch.long),
            "year": torch.tensor([self.year[idx]], dtype=torch.long),
            "http_count": torch.tensor([self.http_count[idx]], dtype=torch.float),
            "diese": torch.tensor([self.diese[idx]], dtype=torch.float),
            "nb_mots": torch.tensor([self.nb_mots[idx]], dtype=torch.float)
        }
        if self.title is not None:
            value["title"] = self.title[idx]
        if hasattr(self, "targets"):
            value["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        if hasattr(self, "class_targets"):
            value["class_target"] = torch.tensor(self.class_targets[idx], dtype=torch.int64)
        return value

# Exemple de collate_fn à utiliser avec DataLoader pour CLIP :
def collate_fn_clip(batch):
    out = {}
    for key in batch[0]:
        if key in ["description", "title"]:
            out[key] = [item[key] for item in batch]  # liste de str
        elif key == "id":
            out[key] = [item[key] for item in batch]
        else:
            out[key] = torch.stack([item[key] for item in batch])
    return out