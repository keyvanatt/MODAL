import torch
import wandb
import hydra
import csv
from tqdm import tqdm
from models.multimodalAttention import *
import os
import matplotlib.pyplot as plt
import torchvision
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import matplotlib
matplotlib.use('Agg')
@hydra.main(config_path="configs", config_name="train")
def main (cfg):
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    train_loader = datamodule.train_dataloader()
    print("Train dataloader created with batch size:", datamodule.batch_size)
    for i,batch in enumerate(train_loader):

        images = batch['image'] if isinstance(batch, dict) and 'image' in batch else batch[0]
        grid = torchvision.utils.make_grid(images)
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.savefig(f'train_batch_{i}.png', bbox_inches='tight')
        plt.show()
        plt.close()

if __name__ == "__main__":
    main()