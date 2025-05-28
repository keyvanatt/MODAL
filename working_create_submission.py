import torch
import wandb
import hydra
import csv
from tqdm import tqdm
from models.multimodalAttention import *


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
    regressor = torch.load("/Data/checkpoints/MIN_ATT&DAR_MULTIMODAL_2025-05-28_18-16-29.pt",weights_only=False)["model_state_dict"]
    model.load_state_dict(regressor, strict=False)
    model.to(device)


    print("Model architecture:")
    print(model)


    model.eval()

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    test_loader = datamodule.test_dataloader()

    resultats = []

    with torch.no_grad():
        # Attention : les DataLoader ne sont pas indexables directement
        for batch in tqdm(test_loader):
            batch["image"] = batch["image"].to(device)
            batch["channel"] = batch["channel"].to(device)
            batch["year"] = batch["year"].to(device)
            batch["http_count"] = batch["http_count"].to(device)
            batch["diese"] = batch["diese"].to(device)
            batch["nb_mots"] = batch["nb_mots"].to(device)
            with torch.no_grad():
                out_data = model(batch)
                out_data = torch.expm1(out_data)
            resultats.append({"ID" : batch["id"].detach().cpu().numpy()[0], "TARGET" : out_data.detach().cpu().numpy()[0][0]})
            
    
    with open('resultat.csv', 'w', newline='') as csvfile:
        fieldnames = ['ID', 'TARGET']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(resultats)


    print("Results saved to resultat.csv")

if __name__ == "__main__":
    main ()
