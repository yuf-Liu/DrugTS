import torch
import torch.nn as nn
import torch.nn.functional as F

    
class IC50Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1)  # ✅ 改这里
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # [B]


class GeneAttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: [B, G, H]
        weights = torch.softmax(self.attn(x), dim=1)  # [B, G, 1]
        out = (x * weights).sum(dim=1)  # [B, H]
        return out


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, ffn_expansion=4, dropout=0.1, batch_first=True):
        super().__init__()
        assert hidden_dim % n_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
        # MultiheadAttention supports batch_first in modern pytorch
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads,
                                                dropout=dropout, batch_first=batch_first)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_expansion, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, queries, memory):
        # queries: [B, gene_dim, H], memory: [B, mem_len, H] (mem_len often =1)
        # cross attention expects (batch_first=True)
        attn_out, _ = self.cross_attn(queries, memory, memory)  # queries attend to memory
        out1 = self.norm1(queries + attn_out)
        ffn_out = self.ffn(out1)
        out2 = self.norm2(out1 + ffn_out)
        return out2


class CrossAttentionDecoderEfficient(nn.Module):
    def __init__(self, cond_dim, gene_dim, n_heads=4, n_layers=1, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gene_dim = gene_dim

        # self.z_to_q = nn.Linear(latent_dim, hidden_dim)
        self.gene_pos_embed = nn.Parameter(torch.randn(gene_dim, hidden_dim) * 0.02)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        assert self.cond_proj.out_features == hidden_dim

        self.ctrl_proj = nn.Linear(1, hidden_dim, bias=False)

        self.layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, n_heads, ffn_expansion=4, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, cond_embed, gene_ctrl=None):
        """
        z: [B, latent_dim]
        cond_embed: [B, cond_dim]
        gene_ctrl: [B, gene_dim] or None
        """
        # B = z.size(0)
        # z_proj = self.z_to_q(z)  # [B, H]

        base_embed = self.gene_pos_embed.unsqueeze(0)  # [1, G, H]

        if gene_ctrl is not None:
            gene_ctrl_expanded = gene_ctrl.unsqueeze(-1)  # [B, G, 1]
            ctrl_embed = self.ctrl_proj(gene_ctrl_expanded)  # [B, G, H]
            gene_embed = base_embed + ctrl_embed
        else:
            gene_embed = base_embed

        queries = gene_embed  # [B, G, H]
        memory = self.cond_proj(cond_embed).unsqueeze(1)  # [B, 1, H]

        out = queries
        for layer in self.layers:
            out = layer(out, memory)  # [B, G, H]
        gene_features = out
        delta = self.out(out).squeeze(-1)  # [B, G]
        return delta, gene_features

class DrugEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, dropout=0.2):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        # x: [B, input_dim]
        return self.net(x)


residual_predict = True

class MainModel(nn.Module):
    def __init__(self, gene_dim, in_smiles_coding_dim, cell_type_num, cell_type_dim, hidden_dim=512, predict_ic50=False):
        super().__init__()
        self.predict_ic50 = predict_ic50
        out_smiles_coding_dim = 1024
        self.cell_type_embed = nn.Embedding(cell_type_num, cell_type_dim)
        self.drug_encoder = DrugEncoder(input_dim=in_smiles_coding_dim, output_dim=out_smiles_coding_dim, hidden_dim=hidden_dim)
        self.cond_dim = cell_type_dim + out_smiles_coding_dim

        self.decoder_pert = CrossAttentionDecoderEfficient(
            cond_dim=self.cond_dim,
            gene_dim=gene_dim,
            n_heads=4, n_layers=1, hidden_dim=hidden_dim
        )

        # ===== IC50 Head =====
        if self.predict_ic50:
            ic50_input_dim = hidden_dim + self.cond_dim  
            self.pool = GeneAttentionPooling(hidden_dim)

            self.ic50_head = IC50Classifier(
                input_dim=ic50_input_dim,
                hidden_dim=512
            )
    
    def forward(self, gene_ctrl, mol_fp, cell_line):
        
        mol_embed = self.drug_encoder(mol_fp)  # [B, cond_dim]
        cell_type_emd = self.cell_type_embed(cell_line)
        cond = torch.cat([cell_type_emd, mol_embed], dim=-1)

        delta, gene_features = self.decoder_pert(cond, gene_ctrl)

        if residual_predict and gene_ctrl is not None:
            gene_pred = gene_ctrl + delta, mol_fp #, cond
        else:
            gene_pred = delta, mol_fp #, cond
        
        if self.predict_ic50:
            gene_summary = self.pool(gene_features)  # [B, H]

            ic50_input = torch.cat([gene_summary, cond], dim=-1)
            logits = self.ic50_head(ic50_input)  # [B, 3]

            return logits

        return gene_pred, mol_fp


def mse_losses(gene_ctrl, gene_pred, gene_pert, residual_predict=True):
    if residual_predict:
        delta_true = gene_pert - gene_ctrl
        delta_pred = gene_pred - gene_ctrl
        delta_loss = F.mse_loss(delta_pred, delta_true, reduction='mean')
    
    pert_loss = F.mse_loss(gene_pred, gene_pert, reduction='mean')
    # pert_loss = F.mse_loss(gene_pred, gene_pert, reduction='mean') + 0.1 * contrast_loss

    return pert_loss, delta_loss


def pearson_corr_loss(pred, target, eps=1e-8):
    """
    Pearson correlation loss = 1 - corr(pred, target)
    """
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    corr_num = torch.sum(pred * target, dim=1)
    corr_den = torch.sqrt(torch.sum(pred ** 2, dim=1) * torch.sum(target ** 2, dim=1) + eps)
    corr = corr_num / (corr_den + eps)
    return 1 - corr.mean()


def ic50_bce_loss(logits, labels):
    """
    logits: [B] or [B,1]
    labels: [B]（0/1）
    """
    logits = logits.view(-1)
    labels = labels.float().view(-1)

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, labels)

    return loss


