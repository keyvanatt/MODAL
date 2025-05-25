import torch
import wandb
import hydra
import csv
from tqdm import tqdm
from models.multimodalAttention import MultiModalAttentionRegressor


@hydra.main(config_path="configs", config_name="train")
def main (cfg):
    test_model(cfg)


def test_model (cfg) :
    """
    L'objectif ici est de récupèrer le modèle enregistré et de le tester sur le dataset test
    et de calculer l'accuracy
    ATTENTION : il faut changer le lien vers le modèle
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiModalAttentionRegressor()
    checkpoint = torch.load("checkpoints/MIN_ATT&DAR_MULTIMODAL_2025-05-25_10-35-48.pt",weights_only=False)["model_state_dict"]
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)

    print("Model architecture:")
    print(model)


    model.eval()

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    test_loader = datamodule.test_dataloader()

    resultats = []

    with torch.no_grad():
        # Attention : les DataLoader ne sont pas indexables directement
        for batch in test_loader:
            batch["image"] = batch["image"].to(device)
            batch["channel"] = batch["channel"].to(device)
            batch["year"] = batch["year"].to(device)
            with torch.no_grad():
                out_data = model(batch)
                #out_data = torch.expm1(out_data)
                # J'ai passé les log1views entre 0 et 1
                out_data = torch.expm1(out_data)
            resultats.append({"ID" : batch["id"].detach().cpu().numpy()[0], "TARGET" : out_data.detach().cpu().numpy()[0][0]})
            
    
    with open('resultat.csv', 'w', newline='') as csvfile:
        fieldnames = ['ID', 'TARGET']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(resultats)


    print("Results saved to resultat.csv")
    print("Computing val loss...")
    # Compute validation loss on validation set
    val_loader = datamodule.val_dataloader()
    total_loss = 0.0
    total_samples = 0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for batch in val_loader:
            batch["image"] = batch["image"].to(device)
            batch["channel"] = batch["channel"].to(device)
            batch["year"] = batch["year"].to(device)
            labels = batch["target"].to(device)
            outputs = model(batch).view(-1)
            loss = criterion(outputs, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    print(f"Validation loss (MSE): {avg_loss:.4f}")

if __name__ == "__main__":
    main ()
