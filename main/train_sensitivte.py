import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import anndata as ad
import numpy as np
import pandas as pd
import random
import os

from CA_model_ic50 import *
from utils import compute_metrics_classify, DrugPredictDataset_sensitive


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class Drug:
    def __init__(self, pert_data, device='cuda'):

        self.device = device

        self.dataloader = pert_data.dataloader
        self.num_genes = len(pert_data.gene_names)
        self.genes = pert_data.gene_names
        self.in_smiles_coding_dim = 1024
        self.cell_type_num = len(pert_data.cell_type)
        self.cell2id = pert_data.cell2id
        self.cell_type_dim = 64
        self.hidden_dim = 256
        self.seed = pert_data.seed

        self.adata = pert_data.adata

        self.save_best_model = None


    def model_initialize(self, **kwargs):
        self.model = MainModel(gene_dim=self.num_genes, in_smiles_coding_dim=self.in_smiles_coding_dim,
                                    cell_type_num=self.cell_type_num, cell_type_dim=self.cell_type_dim,
                                    hidden_dim=self.hidden_dim, predict_ic50=True)

        self.model = self.model.to(self.device)
        # print('Model total Param')
        # summary(self.model) 

    def collect_logits(self, loader):
        self.model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                logits = self.model(
                    batch.x,
                    batch.smiles_coding.squeeze(1),
                    batch.cell_type_ids
                ).squeeze(-1)

                all_logits.append(logits.cpu())
                all_labels.append(batch.ic50.cpu())

        return torch.cat(all_logits), torch.cat(all_labels)

    def collect_logits(self, loader):
        self.model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                logits = self.model(
                    batch.x,
                    batch.smiles_coding.squeeze(1),
                    batch.cell_type_ids
                )

                logits = logits.reshape(logits.size(0), -1)
                labels = batch.ic50
                labels = labels.reshape(-1)

                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


    def train(self, epochs, lr):
        device = self.device
        print("Training on device:", device)
        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader.get('val_loader')

        opt_model = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(opt_model, mode='max', factor=0.5, patience=3, verbose=True)

        os.makedirs("models", exist_ok=True)
        best_auprc = 0
        best_threshold = 0.5
        metrics_history = []

        for epoch in range(1, epochs + 1):
            # ==================== TRAIN ====================
            self.model.train()
            all_logits = []
            all_labels = []
            epoch_loss = 0.0

            for step, batch in enumerate(train_loader):
                batch = batch.to(self.device)

                data_ctrl = batch.x
                drug_names = batch.drug_name
                mol_fp = batch.smiles_coding.squeeze(1)
                cell_line = batch.cell_type_ids
                ic50 = batch.ic50

                ic50_pred = self.model(data_ctrl, mol_fp, cell_line).squeeze(-1)
                loss = ic50_bce_loss(ic50_pred, ic50)
                opt_model.zero_grad()
                loss.backward()
                opt_model.step()

                if step % 500 == 0:
                    print(f"step {step}, loss = {loss.item():.4f}")

                epoch_loss += loss.item()

                all_logits.append(ic50_pred.detach().cpu())
                all_labels.append(ic50.detach().cpu())

            avg_loss = epoch_loss / len(train_loader)

            all_logits = torch.cat(all_logits)
            all_labels = torch.cat(all_labels)
            train_metrics = compute_metrics_classify(
                y_true=all_labels,
                logits=all_logits,
                threshold=0.5)

            print(f"\nEpoch [{epoch}/{epochs}]")
            print(f"Train Loss: {epoch_loss:.4f}, Avg Loss: {avg_loss:.4f}")
            print(f"Train AUROC: {train_metrics['roc_auc']:.4f}, ACC: {train_metrics['accuracy']:.4f}")

            # ==================== VALIDATION ====================
            if val_loader is not None:
                val_logits, val_labels = self.collect_logits(val_loader)
                val_metrics = compute_metrics_classify(y_true=val_labels, logits=val_logits)
                print(f"Val AUROC: {val_metrics['roc_auc']:.4f}, "
                    f"Val Recall: {val_metrics['recall']:.4f}, "
                    f"Precision: {val_metrics['precision']:.4f}, "
                    f"F1: {val_metrics['f1']:.4f}"
                    f"ACC: {val_metrics['accuracy']:.4f}")

                self.scheduler.step(val_metrics['roc_auc'])

                if val_metrics['roc_auc'] > best_auprc:
                   best_auprc = val_metrics['roc_auc']
                   torch.save(self.model.state_dict(),
                       f"melanoma/models/seed{self.seed}/ca_model.pth")
                   print(f"Save best model @ epoch {epoch}")
                
            epoch_metrics = {
                'epoch': epoch,
                # ================= TRAIN =================
                'train_accuracy': train_metrics['accuracy'],
                'train_precision': train_metrics['precision'],
                'train_recall': train_metrics['recall'],
                'train_f1': train_metrics['f1'],
                'train_roc_auc': train_metrics['roc_auc'],
                'train_pr_auc': train_metrics['pr_auc'],
                # ================= VAL =================
                'val_accuracy': val_metrics['accuracy'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_f1': val_metrics['f1'],
                'val_roc_auc': val_metrics['roc_auc'],
                'val_pr_auc': val_metrics['pr_auc']
            }
            metrics_history.append(epoch_metrics)

        df_metrics = pd.DataFrame(metrics_history)
        df_metrics.to_csv(f"melanoma/ic50_models_banlance1/seed{self.seed}/drug.csv", index=False)
        print("Metrics saved")

        if 'test_loader' in self.dataloader:
            test_loader = self.dataloader['test_loader']
            test_logits, test_labels = self.collect_logits(test_loader)
            test_metrics = compute_metrics_classify(
                y_true=test_labels,
                logits=test_logits,
                threshold=best_threshold
            )
            print("Test matrix :", test_metrics)


    def load_pretrained(self, model_path, strict=True):
        print(f"Loading pretrained model from: {model_path}")
        state_dict = torch.load(model_path, map_location=self.device)
        is_dp = isinstance(self.model, torch.nn.DataParallel)

        new_state_dict = {}
        for k, v in state_dict.items():
            if is_dp and not k.startswith("module."):
                new_state_dict["module." + k] = v
            elif (not is_dp) and k.startswith("module."):
                new_state_dict[k.replace("module.", "", 1)] = v
            else:
                new_state_dict[k] = v

        try:
            missing, unexpected = self.model.load_state_dict(
                new_state_dict, strict=strict
            )
            print("Pretrained model loaded.")
            if not strict:
                print("Missing keys:", missing)
                print("Unexpected keys:", unexpected)
        except Exception as e:
            print("Failed to load pretrained weights.")
            raise e

        self.model.to(self.device)
        self.model.eval()
        self.best_model = self.model

        if 'test_loader' in self.dataloader and self.dataloader['test_loader'] is not None:
            test_loader = self.dataloader['test_loader']
        else:
            test_loader = self.dataloader.get('val_loader')

        test_logits, test_labels = self.collect_logits(test_loader)
        # Adjustable threshold
        threshold_ = 0.05
        test_metrics = compute_metrics_classify(
            y_true=test_labels,
            logits=test_logits,
            threshold=threshold_
            )
        print(f"Test matrix threshold={threshold_}:", test_metrics)

        ic50_logits = test_logits.detach().cpu().numpy().flatten()
        test_labels = test_labels.detach().cpu().numpy().flatten()
        y_pred = 1 / (1 + np.exp(-ic50_logits))
        df = pd.DataFrame({
            "ic50_true": test_labels,
            "IC50": y_pred
        })
        # df.to_csv(f"mm_cancer/banlance1/seed{self.seed}/pred_test.csv", index=False)


    def predict(self, drug_names, smiles_list, cell_type, w_file, batch_size=64):

        ctrl_adata = self.adata[self.adata.obs["cell_type"] == cell_type]
        cell_type_id = torch.tensor(self.cell2id[cell_type], dtype=torch.long)

        dataset = DrugPredictDataset_sensitive(
            drug_names, smiles_list, cell_type,
            cell_type_id, ctrl_adata, self.device
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,   
            num_workers=10,
            pin_memory=False
        )

        model = self.best_model
        model.eval()

        all_pred = []
        all_drug_names = []
        all_smiles = []

        with torch.no_grad():
            for i, batch in enumerate(loader):

                batch = batch.to(self.device)
                ctrl_x = batch.x
                mol_fp = batch.smiles_coding
                cell_ids = batch.cell_type_ids

                ic50_logits = model(ctrl_x, mol_fp, cell_ids)
                ic50_logits = ic50_logits.detach().cpu().numpy().flatten()
                y_pred = 1 / (1 + np.exp(-ic50_logits))

                start = i * loader.batch_size
                end = start + batch.num_graphs
                meta_batch = dataset.meta[start:end]

                for j, (drug_name, smiles) in enumerate(meta_batch):
                    all_drug_names.append(drug_name)
                    all_smiles.append(smiles)
                    all_pred.append(y_pred[j])

        df = pd.DataFrame({
            "drug_name": all_drug_names,
            "smiles": all_smiles,
            "IC50": all_pred
        })

        df.to_csv(w_file, index=False)
        print(f"The prediction results have been saved to: {w_file}")
