import torch
import wandb
import hydra
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from models.multimodalAttention import MultiModalAttentionRegressor
from models.dinov2 import DinoV2Finetune

from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.sanity import show_images
import numpy as np


@hydra.main(config_path="configs", config_name="train")
def main (cfg):
    print ("START")
    model = train (cfg)
    #test_model(cfg, model)



def train(cfg, train_idx=None, val_idx=None):
    """
        Fonction globale qui entraîne le réseau. Les paramètres à fixer pour le modèle sont dans le 
        fichier config/train.yaml. 
        
        Ces données config sont accessibles via le paramètre cfg qui n'est
        pas à renseigner lors de l'appel de la fonction.
    """

    logger = (
        wandb.init(
            project="challenge_CSC_43M04_EP",
            name=cfg.experiment_name,
        )
        if cfg.log
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # On crée le modèle défini dans train.yaml sur hydra et le to(device) le balance 
    # sur le cpu s'il existet
    #model = hydra.utils.instantiate(cfg.model.instance).to(device)
    model = MultiModalAttentionRegressor().to(device)
    # On crée l'optimizer défini sur train.yaml
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    # Idem et le datamodule permet globalement de charger les images et les fournir au modèle
    datamodule = hydra.utils.instantiate(cfg.datamodule, train_idx=train_idx, val_idx=val_idx)
    train_loader = datamodule.train_dataloader_dynamique()
    val_loader = datamodule.val_dataloader()
    extreme_train_loader = datamodule.extreme_train_dataloader()
    
    # Permet de charger le modèle avec le meilleur validation loss en cas 
    # de remontée du val_loss
    val_loss_min = np.inf


    # Le scheduler permet de réduire le learning rate en même temps que la loss du validation set diminue
    # mode = 'min' car on veut que le learning rate diminue
    # factor = 0.3 -> le learning rate est diminué de ce facteur quand la condition est remplie
    # patience = 3 : nombre d'epoch sans amélioration avant que le learning rate ne soit réduit
    # Les paramètres sont empiriques. Ne pas hésiter à modifier
    # On retient dans une variable min_learning rate le learning rate final du scheduler
    # Cette variable min_learning_rate est très importante car elle conditionne la fin de
    # la convergence
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=cfg.factor_learning_rate, patience=cfg.patience_learning_rate, min_lr=cfg.min_learning_rate)

    # Envoie le sanity check a wandb pour le training set
    train_sanity = show_images(train_loader, name="assets/sanity/train_images")
    (
        logger.log({"sanity_checks/train_images": wandb.Image(train_sanity)})
        if logger is not None
        else None
    )
    # Envoie le sanity check a wandb pour le validation set
    val_sanity = show_images(val_loader, name="assets/sanity/val_images")
    logger.log(
        {"sanity_checks/val_images": wandb.Image(val_sanity)}
    )
    
    # Helper to plot and log target distributions
    def log_target_distribution(loader, name, logger):
        all_targets = []
        for batch in loader:
            targets = batch["target"].cpu().numpy().flatten()
            all_targets.append(targets)
        all_targets = np.concatenate(all_targets)
        plt.figure()
        plt.hist(all_targets, bins=30, alpha=0.7)
        plt.title(f"Target Distribution: {name}")
        plt.xlabel("Target")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"assets/sanity/{name}_target_dist.png")
        if logger is not None:
            logger.log({f"sanity_checks/{name}_target_dist": wandb.Image(plt.gcf())})
        plt.close()

    log_target_distribution(train_loader, "train", logger)
    log_target_distribution(val_loader, "val", logger)
    # Le max_epoch est juste une sécurité et ne devrait pas influer sur la fin de la convergence
    max_epochs = cfg.max_epochs
    epoch = 0

    ##################
    # Enregistrement #
    ##################

    
    if False:
    #Uniquement si on souhaite restaurer un modèle qui était en entrainement    
        checkpoint = torch.load('checkpoints/min_ATT&DAR_MULTIMODAL_2025-05-21_16-03-10.pt', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch'] + 1  # Reprend à l'epoch suivante
    
    print ("*********")
    print ("Début training loop")
    print ("********")

    # On interrompt la boucle en fonction du learning rate et du max_epoch. 
    # (cf min_learning_rate) plus haut
    # Cf condition break à la fin.
    while True : 
        #################
        # Training loop #
        #################



        train_loader = datamodule.train_dataloader_dynamique(epoch = epoch)


        # Envoie le sanity check a wandb pour le training set
        train_sanity = show_images(train_loader, name="assets/sanity/train_images")
        (
            logger.log({"sanity_checks/train_images_epoch_"+str(epoch): wandb.Image(train_sanity)})
            if logger is not None
            else None
        )

    
        model.train()
        epoch_train_loss = 0
        # Compte le nombre d'images entraînées pour faire la moyenne pour le train_loss
        num_samples_train = 0
        # Là c'est juste la barre de progression pour la console
        if optimizer.param_groups[0]["lr"] > cfg.switch_learning_rate:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        else:
            pbar = tqdm(extreme_train_loader, desc=f"Epoch {epoch} - extreme", leave=False)
        for i, batch in enumerate(pbar):
            # On envoie les images dans le GPU (si dispo)
            batch["image"] = batch["image"].to(device)
            # Pareil pour les labels
            batch["target"] = batch["target"].to(device).squeeze()
            batch["channel"] = batch["channel"].to(device)
            batch["year"] = batch["year"].to(device)
            # Pass forward
            preds = model(batch).squeeze()
            loss = loss_fn(preds, batch["target"])

            # Log weights, biases, and gradients to wandb
            if logger is not None:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        logger.log({f"weights/{name}": wandb.Histogram(param.detach().cpu().numpy())})
                        if param.grad is not None:
                            logger.log({f"grads/{name}": wandb.Histogram(param.grad.detach().cpu().numpy())})
            
            # Partie éliminée pour gagner en vitesse 
            # Là on envoit les données a wandb pour qu'il les affiche
            (
                logger.log({"loss": loss.detach().cpu().numpy(),"year_avg": batch["year"].cpu().numpy().mean()})
                if logger is not None
                else None
            )
            # Classico
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.detach().cpu().numpy() * len(batch["image"])
            num_samples_train += len(batch["image"])
            # Affiche la progression dans la console
            pbar.set_postfix({"train/loss_step": loss.detach().cpu().numpy(), "learning_rate": optimizer.param_groups[0]["lr"]})
        epoch_train_loss /= num_samples_train
        # Pareil, on envoit a wandb
        (
            logger.log(
                {
                    "epoch": epoch,
                    "train/loss_epoch": epoch_train_loss,
                }
            )
            if logger is not None
            else None
        )

        ###################
        # Validation loop #
        ###################

        val_metrics = {}
        epoch_val_loss = 0
        num_samples_val = 0
        model.eval()
        all_targets = []
        all_preds = []
        all_losses = []
        for _, batch in enumerate(val_loader):
            batch["image"] = batch["image"].to(device)
            batch["target"] = batch["target"].to(device).squeeze()
            batch["channel"] = batch["channel"].to(device)
            batch["year"] = batch["year"].to(device)
            with torch.no_grad():
                preds = model(batch).view(-1)
            loss = torch.nn.functional.mse_loss(preds, batch["target"], reduction='none')  # shape: (batch_size,)
            # Collect for scatter plot
            all_targets.append(batch["target"].detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())
            all_losses.append(loss.detach().cpu().numpy())
        
        # After validation loop, scatter predictions vs target
        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)
        all_losses = np.concatenate(all_losses)
        epoch_val_loss = all_losses.mean()

        # Plot 1: Predictions vs Target
        plt.figure(figsize=(6, 5))
        plt.scatter(all_targets, all_preds, alpha=0.5,c=all_losses, cmap='viridis')
        plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--')
        plt.xlabel("Target")
        plt.ylabel("Prediction")
        plt.title("Predictions vs Target (Validation)")
        plt.tight_layout()
        plt.savefig("assets/sanity/val_pred_vs_target.png")
        if logger is not None:
            logger.log({f"predictions/val_pred_vs_target_epoch_{epoch:03d}": wandb.Image(plt.gcf())})
        plt.close()

        # On envoie la loss au scheduler pour qu'il puisse influencer le learning rate
        scheduler.step(epoch_val_loss)
        # On récupère le learning rate effectif du modèle avant de l'envoyer à wandb
        current_lr = optimizer.param_groups[0]["lr"]
        print ("Epoch : " + str(epoch) + ", Learning rate : " + str(current_lr)+ ", Training Loss : " + str(epoch_train_loss) + ", Validation Loss : " + str(epoch_val_loss))
        # On envoie tout à wandb
        val_metrics["val/loss_epoch"] = epoch_val_loss
        val_metrics["learning_rate"] = current_lr
        (
            logger.log(
                {
                    "epoch": epoch,
                    **val_metrics,
                }
            )
            if logger is not None
            else None
        )

        ##############
        # Sauvegarde #
        ##############

        if (epoch == 0) :   
            # 1ere epoch : on n'a pas encore enregistré de modèle
            val_loss_min = epoch_val_loss
        
        # On sauvegarde à intervalles réguliers
        if (epoch % cfg.checkpoint_interval == 0) :
            # Si oui, on sauvegarde le modèle    
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': epoch_val_loss,
            }
            torch.save(checkpoint, cfg.checkpoint_path)
            print ("Modèle enregistré !")

        if (epoch_val_loss <= val_loss_min) :    
            val_loss_min = epoch_val_loss
            # Si oui, on sauvegarde le modèle    
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': epoch_val_loss,
            }
            torch.save(checkpoint, cfg.min_checkpoint_path)
            print ("Modèle optimal enregistré !")


        # Si le modèle est pire que le meilleur, on repart avec le précédent
        if (epoch != 0 and val_loss_min < epoch_val_loss * (1 - cfg.aberration_val_loss)):
            print ("Aberration de la val_loss")
            checkpoint = torch.load(cfg.min_checkpoint_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for param_group in optimizer.param_groups:
                param_group['lr'] *= cfg.factor_learning_rate
        

        ################################
        # Conditions de sortie de loop #
        ################################
        epoch += 1

        if (current_lr <= cfg.min_learning_rate or epoch > max_epochs) :
            print ("***")
            print ("Fin de la convergence")
            print ("***")
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
    print ("Modèle enregistré !")
    return (model)


def test_model (cfg, model) :
    """
    Non fonctionnel pour l'instant cf le fichier séparé
    L'objectif ici est de tester le modèle sur le dataset test et de calculer l'accuracy
    du modèle
    """

    model.eval()

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    test_loader = datamodule.test_dataloader()

    with torch.no_grad():
        out_data = model(test_loader[0][1])
    
    print (out_data)



if __name__ == "__main__":
    main ()
