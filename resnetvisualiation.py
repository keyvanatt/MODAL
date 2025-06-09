import torch
import torch.nn.functional as F
from torchvision.models import resnet50
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Paramètres
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Préparer modèle
model = resnet50(pretrained=True).to(device).eval()

# 2. Hook pour récupérer la sortie de la dernière feature map
features = []
def hook_fn(module, input, output):
    features.clear()
    features.append(output)  # shape: (1, 2048, 7, 7)

hook_handle = model.layer4.register_forward_hook(hook_fn)

# 3. Charger et transformer l'image de base
img = Image.open("dataset/test/2.jpg").convert("RGB")
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 4. Normalisation inverse pour affichage
def deprocess(tensor):
    tensor = tensor.detach().cpu().squeeze()
    tensor = tensor.permute(1, 2, 0)  # C, H, W -> H, W, C
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    return tensor.numpy()

# 5. Générer les 49 images maximisées
all_imgs = []
for target_index in range(49):
    print(f"Maximizing feature map vector {target_index + 1}/49")
    img_tensor = torch.zeros((1, 3, 224, 224), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([img_tensor], lr=0.05, weight_decay=1e-6)

    for step in range(200):
        optimizer.zero_grad()
        model(img_tensor)
        fmap = features[0]  # shape: (1, 2048, 7, 7)
        fmap_flat = fmap.view(1, 2048, -1).permute(0, 2, 1)  # (1, 49, 2048)
        target_vector = fmap_flat[0, target_index]  # (2048,)
        loss = -target_vector.norm(p=2)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            img_tensor.clamp_(-1.5, 1.5)
    final_img = deprocess(img_tensor)
    all_imgs.append(final_img)

# 6. Assembler dans une grille 7x7
fig, axes = plt.subplots(7, 7, figsize=(14, 14))
for idx, ax in enumerate(axes.flat):
    ax.imshow(all_imgs[idx])
    ax.set_title(f"{idx}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("maximized_vectors_grid.png")
plt.show()
