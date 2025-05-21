import torch
import torch.nn as nn


class DinoV2Finetune(nn.Module):
    def __init__(self, frozen=True, regression=False, get_tokens=False):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone.head = nn.Identity()
        self.dim = self.backbone.norm.normalized_shape[0]
        if not frozen:
            for name, param in list(self.backbone.named_parameters())[-2:]:
                param.requires_grad = True

        self.get_tokens = get_tokens

        self.lsh_bits = 16
        # Matrice de projection LSH (fixée pour tout le modèle)
        self.register_buffer("lsh_proj", torch.randn(self.dim, self.lsh_bits))


    """
    Précédente version sans LSH

    def forward(self, x,k=32):
        if self.get_tokens:
            patch_tokens = self.backbone.get_intermediate_layers(x["image"], n=1)[0]
            scores = patch_tokens.norm(dim=-1)  # (B, N)
            _, idx = scores.topk(k=k, dim=-1)  # on garde les 32 plus informatifs
            # Gather top-k tokens
            idx_expanded = idx.unsqueeze(-1).expand(-1, -1, patch_tokens.size(-1))  # (B, 32, dim)
            x = torch.gather(patch_tokens, 1, idx_expanded)  # (B, 32, dim)
            #x = torch.stack([patch_tokens[b, idx[b]] for b in range(B)], dim=0)  
        else:
            x = self.backbone(x["image"])
        return x
    """

    def forward(self, x, k=32):
        """
        Implémente la projection LSH à partir du code de Kayvan
        """
        if self.get_tokens:
            # Je récupère les tokens
            patch_tokens = self.backbone.get_intermediate_layers(x["image"], n=1)[0]  # (B, N, dim)
            # Projection LSH
            # On projète les tokens sur des matrices aléatoires de taille (dim, lsh_bits)
            # > 0 : on applique un seuil à 0 pour obtenir le code binaire (hash)
            hashes = (patch_tokens @ self.lsh_proj > 0).int()  # (B, N, lsh_bits)
            # On calcul les identifiants de hachage
            # Chaque code binaire est converti en un entier unique : chaque bit du hash est multiplié par une puissance de 2
            # device = ... : on crée le tenseur directement sur le gpu
            hash_ids = torch.sum(hashes * (2 ** torch.arange(self.lsh_bits, device=patch_tokens.device)), dim=-1)  # (B, N)
            # Pour chaque batch, on choisit k tokens des plus gros buckets
            idx = []
            for b in range(patch_tokens.size(0)):
                # Unique contient tous les id différents 
                # Count indique combien de tokens de l'image b ont ce hach id
                unique, counts = hash_ids[b].unique(return_counts=True)
                # On trie les ids en fontion du plus populaire
                top_buckets = unique[counts.argsort(descending=True)[:k]]
                # On fait un masque booléen (très classe bravo Louis)
                mask = torch.isin(hash_ids[b], top_buckets)
                # On récupère les indices des tokens sélectionnés
                selected = torch.nonzero(mask).squeeze()
                # Si il y a plus que k tokens (probable) on choisit au hasard. 
                # Sinon on complète par des 0
                if selected.numel() > k:
                    selected = selected[torch.randperm(selected.numel())[:k]]
                elif selected.numel() < k:
                    pad = selected.new_full((k - selected.numel(),), selected[0] if selected.numel() > 0 else 0)
                    selected = torch.cat([selected, pad])
                idx.append(selected)
            idx = torch.stack(idx, dim=0)  # (B, k)
            idx_expanded = idx.unsqueeze(-1).expand(-1, -1, patch_tokens.size(-1))  # (B, k, dim)
            x = torch.gather(patch_tokens, 1, idx_expanded)  # (B, k, dim)
        else:
            x = self.backbone(x["image"])
        return x

    def forward2(self, x, k=32):
        """
        Implémente PCA à partir du code de Kayvan
        """
        if self.get_tokens:
            patch_tokens = self.backbone.get_intermediate_layers(x["image"], n=1)[0]  # (B, N, dim)
            B, _, _ = patch_tokens.shape
            idx = []
            for b in range(B):
                tokens = patch_tokens[b]  # (N, D)
                # Centrage
                tokens_centered = tokens - tokens.mean(dim=0, keepdim=True)
                # Matrice de covariance
                cov = tokens_centered.t() @ tokens_centered  # (D, D)
                # Vecteurs propres
                _, vecProp = torch.linalg.eigh(cov)
                # Première composante principale (plus grande valeur propre)
                pc1 = vecProp[:, -1]  # (D,)
                # Projeter les tokens sur la première composante
                projections = tokens_centered @ pc1  # (N,)
                # Prendre les k tokens avec la plus grande valeur absolue de projection
                _, top_idx = torch.topk(projections.abs(), k)
                idx.append(top_idx)
            idx = torch.stack(idx, dim=0)  # (B, k)
            # On récupère les tokens
            idx_expanded = idx.unsqueeze(-1).expand(-1, -1, patch_tokens.size(-1))  # (B, k, dim)
            x = torch.gather(patch_tokens, 1, idx_expanded)  # (B, k, dim)
        else:
            x = self.backbone(x["image"])
        return x




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DinoV2Finetune(frozen=False, regression=False, get_tokens=True).to(device)
    
    image_tensor = torch.randn(2, 3, 224, 224).to(device)
    x = {"image": image_tensor}
    embeddings = model(x)
    print(embeddings.shape) 