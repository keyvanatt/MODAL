import torch
import wandb
import hydra
import csv
from tqdm import tqdm
from models.multimodalAttention import *
import os
import numpy as np
import matplotlib.pyplot as plt

def contains_bengali(text):
    return any('\u0980' <= c <= '\u09FF' for c in text)

@hydra.main(config_path="configs", config_name="train")
def main (cfg):
    attention_weights, tokenized_texts,images =  visualisation(cfg)
    fig, axs = plt.subplots(len(attention_weights), 1, figsize=(12, 3 * len(attention_weights)))

    if len(attention_weights) == 1:
        axs = [axs]

    for i, (attn, tokens) in enumerate(zip(attention_weights, tokenized_texts)):
        tokens.append("[SEP]")  # Ajouter le token de fin
        
        im = axs[i].imshow(attn, aspect='auto', cmap='viridis')
        axs[i].set_yticks(range(len(tokens)))
        axs[i].set_yticklabels(tokens)
        axs[i].set_xticks(range(49))
        axs[i].set_xticklabels(range(49))
        fig.colorbar(im, ax=axs[i])


    plt.tight_layout()
    plt.savefig("attention_weights_visualization.png")
    plt.show()
    print("Visualisation des poids d'attention terminée.")
    # Calcul du softmax pour chaque vecteur d'attention
    # Afficher les softmax sur les mêmes sous-figures que les heatmaps
    fig, axs = plt.subplots(len(attention_weights), 1, figsize=(5, 3 * len(attention_weights)))

    for i, (attn, tokens) in enumerate(zip(attention_weights, tokenized_texts)):
        attn_vec = attn
        softmax_vals = torch.nn.functional.softmax(torch.tensor(attn_vec), dim=0).numpy()
        softmax_vals = softmax_vals.mean(axis=1)  # Normalisation pour chaque exemple

        axs[i].plot(range(len(tokens)), softmax_vals, color='red', alpha=1, label='Softmax')
        axs[i].set_xticks(range(len(tokens)))
        axs[i].set_xticklabels(tokens, rotation=45, ha='right')
        axs[i].set_yticks([])
    plt.tight_layout()
    plt.savefig("attention_title_visualization.png")
    plt.show()
    print("Visualisation des textes d'attention terminée.")

    # Affichage des images associées aux poids d'attention
    fig, axs = plt.subplots(3,2, figsize=(4*2, 4 * 3))
    axs = np.array(axs).flatten()

    if len(images) == 1:
        axs = [axs]
    for i, img in enumerate(images):
        # Dessiner une grille 7x7 sur l'image et numéroter chaque case de 0 à 48
        h, w = img.shape[1], img.shape[2]
        num_rows, num_cols = 7, 7
        cell_h, cell_w = h // num_rows, w // num_cols

        for row in range(num_rows):
            for col in range(num_cols):
                y0, x0 = row * cell_h, col * cell_w
                y1, x1 = (row + 1) * cell_h, (col + 1) * cell_w
                rect = plt.Rectangle((x0, y0), cell_w, cell_h, linewidth=1, edgecolor='red', facecolor='none')
                axs[i].add_patch(rect)
                idx = row * num_cols + col
                axs[i].text(x0 + cell_w // 2, y0 + cell_h // 2, str(idx), color='white', ha='center', va='center', fontsize=8, weight='bold')
        axs[i].imshow(np.transpose(img, (1, 2, 0)))
        axs[i].axis('off')
        axs[i].set_title(f"Image {i+1}")
    axs[-1].axis('off')  # Hide the last unused subplot if there are fewer than 6 images
    plt.tight_layout()
    plt.savefig("attention_images_visualization.png")
    plt.show()
    print("Visualisation des images terminée.")

def visualisation(cfg):
    """
    L'objectif ici est de visualiser les poids d'attention du modèle
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiModalAttentionRegressor()
    regressor = torch.load("/Data/checkpoints/MIN_ATT&DAR_MULTIMODAL_2025-05-29_19-28-20.pt",weights_only=False)["model_state_dict"]
    model.load_state_dict(regressor, strict=False)
    model.to(device)

    print("Model architecture:")
    print(model)

    model.eval()

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    test_loader = datamodule.test_dataloader()

    attention_weights = []
    tokenized_texts = []
    images = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch["image"] = batch["image"].to(device)
            batch["channel"] = batch["channel"].to(device)
            batch["year"] = batch["year"].to(device)
            batch["http_count"] = batch["http_count"].to(device)
            batch["diese"] = batch["diese"].to(device)
            batch["nb_mots"] = batch["nb_mots"].to(device)



            out_data, attention_weights_txt, attention_weights_img = model(batch, return_attention=True)
            tokenized_text = model.text_encoder.tokenizer.tokenize(batch["title"][0])
            print(len(tokenized_text))
            print(attention_weights_txt.shape)

            # Vérifie s'il n'y a pas de caractère spécial ni de caractère non latin (ex: bengali)
            if any(contains_bengali(token) for token in tokenized_text):
                print("Caractère bengali détecté, on skip.")
            else:
                tokenized_texts.append(tokenized_text)
                attention_weights.append(attention_weights_txt[0][:,:].cpu().numpy())
                images.append(batch["image"][0].cpu().numpy())
        
            if len(attention_weights) >= 5:
                return attention_weights[:5], tokenized_texts[:5],images[:5]

if __name__ == "__main__":
    main()
