import torch
import wandb
import hydra
from tqdm import tqdm


from utils.sanity import show_images


@hydra.main(config_path="configs", config_name="train")
def main (cfg):
    model = train (cfg)
    #test_model(cfg, model)



def train(cfg):
    """
        Fonction globale qui entraîne le réseau. Les paramètres à fixer pour le modèle sont dans le 
        fichier config/train.yaml. 
        
        Ces données config sont accessibles via le paramètre cfg qui n'est
        pas à renseigner lors de l'appel de la fonction.
    """

    logger = (
        wandb.init(project="challenge_CSC_43M04_EP", name=cfg.experiment_name)
        if cfg.log
        else None
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # On crée le modèle défini dans train.yaml sur hydra et le to(device) le balance 
    # sur le cpu s'il existe
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    # On crée l'optimizer défini sur train.yaml
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    # Idem et le datamodule permet globalement de charger les images et les fournir au modèle
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
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

    # -- loop over epochs
    for epoch in tqdm(range(cfg.epochs), desc="Epochs"):
        # -- loop over training batches
        model.train()
        epoch_train_loss = 0
        # Compte le nombre d'images entraînées pour faire la moyenne pour le train_loss
        num_samples_train = 0
        # Là c'est juste la barre de progression pour la console
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for i, batch in enumerate(pbar):
            # On envoie les images dans le GPU (si dispo)
            batch["image"] = batch["image"].to(device)
            # Pareil pour les labels
            batch["target"] = batch["target"].to(device).squeeze()
            # Pass forward
            preds = model(batch).squeeze()
            # On calcule le loss. Je sais pas si ca se fait sur le GPU mais à la limite on s'en fout
            loss = loss_fn(preds, batch["target"])
            # Là on envoit les données a wandb pour qu'il les affiche
            (
                logger.log({"loss": loss.detach().cpu().numpy()})
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
            pbar.set_postfix({"train/loss_step": loss.detach().cpu().numpy()})
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

        # -- validation loop
        val_metrics = {}
        epoch_val_loss = 0
        num_samples_val = 0
        model.eval()
        for _, batch in enumerate(val_loader):
            batch["image"] = batch["image"].to(device)
            batch["target"] = batch["target"].to(device).squeeze()
            with torch.no_grad():
                preds = model(batch).squeeze()
            loss = loss_fn(preds, batch["target"])
            epoch_val_loss += loss.detach().cpu().numpy() * len(batch["image"])
            num_samples_val += len(batch["image"])
        epoch_val_loss /= num_samples_val
        val_metrics["val/loss_epoch"] = epoch_val_loss
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

    print(
        f"""Epoch {epoch}: 
        Training metrics:
        - Train Loss: {epoch_train_loss:.4f},
        Validation metrics: 
        - Val Loss: {epoch_val_loss:.4f}"""
    )

    if cfg.log:
        logger.finish()

    torch.save(model.state_dict(), cfg.checkpoint_path)
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
