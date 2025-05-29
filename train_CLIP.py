import torch
import wandb
import hydra
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
from open_clip import create_model_and_transforms

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from models.multimodal_CLIP import MultiModalCLIPRegressor
from data.dataset_CLIP import DatasetCLIP, collate_fn_clip
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.sanity import show_images
from sklearn.model_selection import train_test_split

@hydra.main(config_path="configs", config_name="train")
def main(cfg):
    print("START")
    model = train(cfg)
    # test_model(cfg, model)

def train(cfg, train_idx=None, val_idx=None):
    logger = (
        wandb.init(
            project="challenge_CSC_43M04_EP",
            name=cfg.experiment_name,
        )
        if cfg.log
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###############################
    ##### Nombre min de vues ######
    ###############################
    # ATTENTION : pour les trop grands nombres (a partir de 13??) ca bug et je ne 
    # sais pas trop pourquoi (pas assez de données ?)

    min_views = 12

    # Charger CLIP et son préprocessing
    clip_model, _, clip_preprocess = create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

    # Entrainer le modèle sur un subset e données avec un minimum de views

    # Créer les datasets et dataloaders
    full_dataset = DatasetCLIP(cfg.datamodule.dataset_path, "train_val", transforms=clip_preprocess, min_views=min_views)
    indices = np.arange(len(full_dataset))
    years = np.array(full_dataset.year) 
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=years
    )
    train_set = DatasetCLIP(
        cfg.datamodule.dataset_path,
        "train_val",
        transforms=clip_preprocess,
        indices=train_idx,
        min_views=min_views
    )
    val_set = DatasetCLIP(
        cfg.datamodule.dataset_path,
        "train_val",
        transforms=clip_preprocess,
        indices=val_idx,
        min_views=min_views
    )

    train_loader = DataLoader(train_set, batch_size=cfg.datamodule.batch_size, shuffle=True, num_workers=cfg.datamodule.num_workers, collate_fn=collate_fn_clip)
    val_loader = DataLoader(val_set, batch_size=cfg.datamodule.batch_size, shuffle=False, num_workers=cfg.datamodule.num_workers, collate_fn=collate_fn_clip)

    model = MultiModalCLIPRegressor().to(device)
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=cfg.factor_learning_rate, patience=cfg.patience_learning_rate, min_lr=cfg.min_learning_rate)

    print("Sanity checks...")
    # Sanity check images
    train_sanity = show_images(train_loader, name="assets/sanity/train_images")
    if logger is not None:
        logger.log({"sanity_checks/train_images": wandb.Image(train_sanity)})
    val_sanity = show_images(val_loader, name="assets/sanity/val_images")
    if logger is not None:
        logger.log({"sanity_checks/val_images": wandb.Image(val_sanity)})

    max_epochs = cfg.max_epochs
    epoch = 0
    val_loss_min = np.inf

    while True:
        # --- Training loop ---
        model.train()
        epoch_train_loss = 0
        num_samples_train = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            preds = model(batch).squeeze()
            target = batch["target"].to(device).squeeze()
            loss = loss_fn(preds, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.detach().cpu().numpy() * len(target)
            num_samples_train += len(target)
            pbar.set_postfix({"train/loss_step": loss.detach().cpu().numpy(), "learning_rate": optimizer.param_groups[0]["lr"]})

        epoch_train_loss /= num_samples_train
        if logger is not None:
            logger.log({"epoch": epoch, "train/loss_epoch": epoch_train_loss})

        # --- Validation loop ---
        model.eval()
        epoch_val_loss = 0
        num_samples_val = 0
        all_targets, all_preds, all_val_losses, all_train_losses = [], [], [], []
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}", leave=False):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.no_grad():
                preds = model(batch).view(-1)
            targets = batch["target"].to(device).view(-1)
            loss = torch.nn.functional.mse_loss(preds, targets, reduction='none')
            train_loss = loss_fn(preds, targets, reduction='none')
            num_samples_val += len(targets)
            epoch_val_loss += loss.detach().cpu().numpy().sum()
            all_targets.append(targets.cpu().numpy())
            all_val_losses.append(loss.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_train_losses.append(train_loss.cpu().numpy())

        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)
        all_val_losses = np.concatenate(all_val_losses)
        all_train_losses = np.concatenate(all_train_losses)
        high_val_loss = all_val_losses[all_val_losses > 10].mean() if len(all_val_losses[all_val_losses > 10]) > 0 else 0
        low_val_loss = all_val_losses[all_val_losses < 10].mean() if len(all_val_losses[all_val_losses < 10]) > 0 else 0

        # Plots
        plt.figure(figsize=(6, 5))
        plt.scatter(all_targets, all_preds, alpha=0.5, c=all_val_losses, cmap='viridis')
        plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--')
        plt.xlabel("Target")
        plt.ylabel("Prediction")
        plt.title("Predictions vs Target (Validation)")
        plt.tight_layout()
        if logger is not None:
            logger.log({f"predictions/val_pred_vs_target": wandb.Image(plt.gcf()), "epoch": epoch})
        plt.close()

        plt.figure(figsize=(6, 5))
        sc = plt.scatter(all_targets, all_val_losses, alpha=0.5, c=all_preds, cmap='viridis')
        plt.xlabel("Target")
        plt.ylabel("MSE Loss")
        plt.title("Loss vs Target (Validation)")
        plt.colorbar(sc, label="Prediction")
        plt.tight_layout()
        if logger is not None:
            logger.log({f"predictions/val_loss_vs_target": wandb.Image(plt.gcf()), "epoch": epoch})
        plt.close()

        plt.figure(figsize=(6, 5))
        sc = plt.scatter(all_targets, all_train_losses, alpha=0.5, c=all_preds, cmap='viridis')
        plt.xlabel("Target")
        plt.ylabel("Train Loss")
        plt.title("Train Loss vs Target (Validation)")
        plt.colorbar(sc, label="Prediction")
        plt.tight_layout()
        if logger is not None:
            logger.log({f"predictions/val_train_loss_vs_target": wandb.Image(plt.gcf()), "epoch": epoch})
        plt.close()

        epoch_val_loss /= num_samples_val
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch : {epoch}, Learning rate : {current_lr}, Training Loss : {epoch_train_loss}, Validation Loss : {epoch_val_loss}")

        val_metrics = {
            "val/loss_epoch": epoch_val_loss,
            "learning_rate": current_lr,
            "val/high_loss": high_val_loss,
            "val/low_loss": low_val_loss,
        }
        if logger is not None:
            logger.log({"epoch": epoch, **val_metrics})

        # Sauvegarde
        if epoch == 0:
            val_loss_min = epoch_val_loss
        if epoch % cfg.checkpoint_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': epoch_val_loss,
            }
            torch.save(checkpoint, cfg.checkpoint_path)
            print("Modèle enregistré !")
        if epoch_val_loss <= val_loss_min:
            val_loss_min = epoch_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': epoch_val_loss,
            }
            torch.save(checkpoint, cfg.min_checkpoint_path)
            print("Modèle optimal enregistré !")
        if epoch != 0 and val_loss_min < epoch_val_loss * (1 - cfg.aberration_val_loss):
            print("Aberration de la val_loss")
            checkpoint = torch.load(cfg.min_checkpoint_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for param_group in optimizer.param_groups:
                param_group['lr'] *= cfg.factor_learning_rate

        epoch += 1
        if current_lr <= cfg.min_learning_rate or epoch > max_epochs:
            print("***")
            print("Fin de la convergence")
            print("***")
            break

    print(
        f"""Epoch {epoch}: 
        Training metrics:
        - Train Loss: {epoch_train_loss:.4f},
        Validation metrics: 
        - Val Loss: {epoch_val_loss:.4f}"""
    )

    if cfg.log:
        logger.finish()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(checkpoint, cfg.checkpoint_path)
    print("Modèle enregistré !")
    return model

if __name__ == "__main__":
    main()