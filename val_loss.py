import torch
import wandb
import hydra
import csv
from tqdm import tqdm
from models.multimodalAttention import *
@hydra.main(config_path="configs", config_name="train")
def main (cfg):
    print("Computing val loss...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiModalAttentionRegressor()
    regressor = torch.load("/Data/checkpoints/MIN_ATT&DAR_MULTIMODAL_2025-05-28_18-16-29.pt",weights_only=False)["model_state_dict"]
    model.load_state_dict(regressor, strict=False)
    model.to(device)


    print("Model architecture:")
    print(model)


    model.eval()

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    # Compute validation loss on validation set
    val_loader = datamodule.val_dataloader()
    total_loss = 0.0
    total_samples = 0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for batch in tqdm(val_loader):
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
    main()