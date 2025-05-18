import torch
import wandb
import hydra
import csv
from tqdm import tqdm


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

    model = hydra.utils.instantiate(cfg.model.instance)
    checkpoint = torch.load("/users/eleves-b/2023/keyvan.attarian/MODAL/checkpoints/ATT&DAR_MULTIMODAL_2025-05-14_18-19-08.pt",weights_only=False)
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)


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
                out_data = torch.expm1(out_data)
            resultats.append({"ID" : batch["id"].detach().cpu().numpy()[0], "TARGET" : out_data.detach().cpu().numpy()[0][0]})
            
    
    print (resultats)

    with open('resultat.csv', 'w', newline='') as csvfile:
        fieldnames = ['ID', 'TARGET']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(resultats)



if __name__ == "__main__":
    main ()
