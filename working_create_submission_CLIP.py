import torch
import hydra
import csv
from tqdm import tqdm
from open_clip import create_model_and_transforms
from models.multimodal_CLIP import MultiModalCLIPRegressor
from data.dataset_CLIP import DatasetCLIP, collate_fn_clip

@hydra.main(config_path="configs", config_name="train")
def main(cfg):
    test_model(cfg)

def test_model(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger CLIP et son préprocessing
    clip_model, _, clip_preprocess = create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

    # Charger le modèle entraîné
    model = MultiModalCLIPRegressor()
    checkpoint = torch.load("/Data/checkpoints/ATT&DAR_MULTIMODAL_2025-05-29_02-49-54.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    # Préparer le dataset test
    test_dataset = DatasetCLIP(cfg.datamodule.dataset_path, "test", transforms=clip_preprocess)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=cfg.datamodule.batch_size, shuffle=False, num_workers=cfg.datamodule.num_workers, collate_fn=collate_fn_clip)

    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out_data = model(batch)
            out_data = torch.expm1(out_data)  # Si tu as entraîné sur log1p(views)
            for i in range(len(batch["id"])):
                results.append({
                    "ID": batch["id"][i],
                    "TARGET": out_data[i].item()
                })

    with open('resultat.csv', 'w', newline='') as csvfile:
        fieldnames = ['ID', 'TARGET']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("Results saved to resultat.csv")

if __name__ == "__main__":
    main()