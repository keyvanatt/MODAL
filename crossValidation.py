import torch
import wandb
import hydra
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from train import train
from utils.sanity import show_images
import numpy as np


@hydra.main(config_path="configs", config_name="crossvalidation")
def cross_validation(cfg):
    """
        Fonction globale qui entraîne le réseau. Les paramètres à fixer pour le modèle sont dans le 
        fichier config/crossvalidation.yaml. 
        
        Ces données config sont accessibles via le paramètre cfg qui n'est
        pas à renseigner lors de l'appel de la fonction.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # On crée le modèle défini dans train.yaml sur hydra et le to(device) le balance 
    # sur le cpu s'il existe
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    # On crée l'optimizer défini sur train.yaml
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    # Idem et le datamodule permet globalement de charger les images et les fournir au modèle
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    train_val_dataset = datamodule.full_dataset
    X = np.zeros(len(train_val_dataset), dtype=object) #On s'en fout de X
    y = train_val_dataset.targets #y est important pour le stratifiedKFold
    print(f"y : {y}")
    
    # Envoie le sanity check a wandb pour le training set

    # do the cross validation
    if cfg.stratified:
        kf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=cfg.shuffle, random_state=cfg.seed)
    else:
        kf = KFold(n_splits=cfg.n_splits, shuffle=cfg.shuffle, random_state=cfg.seed)
    avg_val_loss = 0.0
    avg_accuracy = 0.0
    for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X, y), total=cfg.n_splits, desc="Cross Validation Folds")):
        print(f"Fold {fold + 1}/{cfg.n_splits}")
        model = train(cfg,train_idx, val_idx)
        
        val_loss = model.loss
        
        print(f"Fold {fold + 1} - Validation Loss: {val_loss:.4f}")
        

        avg_val_loss += val_loss
    avg_val_loss /= cfg.n_splits
    print(f"Average Validation Loss: {avg_val_loss:.4f}")
    
    
    return avg_val_loss, avg_accuracy
if __name__ == "__main__":
    cross_validation()

