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
    model.load_state_dict(torch.load("/users/eleves-b/2023/keyvan.attarian/MODAL/checkpoints/ATT&DAR_DINOV2_2025-05-13_16-44-40.pt",weights_only=False)['model_state_dict'], strict=False)
    model.to(device)


    model.eval()

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    test_loader = datamodule.test_dataloader()

    resultats = []

    with torch.no_grad():
        # Attention : les DataLoader ne sont pas indexables directement
        for batch in test_loader:
            batch["image"] = batch["image"].to(device)
            with torch.no_grad():
                out_data = model(batch)
            resultats.append({"ID" : batch["id"].detach().cpu().numpy()[0], "TARGET" : out_data.detach().cpu().numpy()[0][0]})
            
    
    print (resultats)

    with open('resultat.csv', 'w', newline='') as csvfile:
        fieldnames = ['ID', 'TARGET']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(resultats)



if __name__ == "__main__":
    main ()
