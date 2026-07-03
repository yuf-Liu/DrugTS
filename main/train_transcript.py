from torch_geometric.loader import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import anndata as ad
import numpy as np
import pandas as pd
import gc
import random
import os

from CA_model import *
from utils import compute_metrics, DrugPredictDataset_transcript


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

        self.ctrl_adata = pert_data.ctrl_adata

        self.save_best_model = None

    def model_initialize(self, **kwargs):

        self.model = MainModel(gene_dim=self.num_genes, in_smiles_coding_dim=self.in_smiles_coding_dim,
                               cell_type_num=self.cell_type_num, cell_type_dim=self.cell_type_dim,
                               hidden_dim=self.hidden_dim, predict_ic50=False)

        self.model = self.model.to(self.device)
        # print('Model total Param')
        # summary(self.model) 

    def train(self, epochs, lr):
        device = self.device
        print("Training on device:", device)
        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader.get('val_loader')

        opt_model = torch.optim.Adam(self.model.parameters(), lr=lr)

        os.makedirs("models", exist_ok=True)

        loss = np.inf
        for epoch in range(1, epochs+1):
            self.model.train()
            epoch_loss = 0.0

            for step, batch in enumerate(train_loader):
                batch = batch.to(self.device)
                data_ctrl = batch.x
                data_pert = batch.y
                mol_fp = batch.smiles_coding
                cell_line = batch.cell_type_ids

                pert_pred, cond = self.model(data_ctrl, mol_fp, cell_line)

                delta_true = data_pert - data_ctrl
                delta_pred = pert_pred - data_ctrl
                # cos_similar = F.cosine_similarity(delta_pred, delta_true, dim=1).mean().item()

                mse_loss, delta_loss = mse_losses(data_ctrl, pert_pred, data_pert)
                pearson_loss = pearson_corr_loss(pert_pred, data_pert)

                # Total loss
                total_loss = (mse_loss + pearson_loss)

                opt_model.zero_grad()
                total_loss.backward()
                opt_model.step()
                if step % 100 == 0:
                    print(f"[epoch {epoch} step {step}]: "
                          f"mse_loss={mse_loss.item():.4f}, pearson_loss={pearson_loss.item():.4f}, total_loss={total_loss.item():.4f}")
                epoch_loss += total_loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch}/{epochs}]:" 
                  f"Train Loss: {epoch_loss}, AvgG_Loss: {avg_loss:.4f}")
            
            tr_mse, tr_r2, tr_pcc, tr_sp, tr_delta_r2, tr_delta_pcc, tr_delta_sp \
                = self.evaluate_model(self.model, train_loader, self.device)
            print(f"Train matrix (MSE, R2, PCC, SP, delta_R2, delta_PCC, delta_SP): "
                  f"{tr_mse:.4f}, {tr_r2:.4f}, {tr_pcc:.4f}, {tr_sp:.4f}, {tr_delta_r2:.4f}, {tr_delta_pcc:.4f}, {tr_delta_sp:.4f}")

            if val_loader is not None:
                val_mse, val_r2, val_pcc, val_sp, val_delta_r2, val_delta_pcc, val_delta_sp \
                    = self.evaluate_model(self.model, val_loader, self.device)
                print(f"Val matrix (MSE, R2, PCC, SP, delta_R2, delta_PCC, delta_SP): "
                      f"{val_mse:.4f}, {val_r2:.4f}, {val_pcc:.4f}, {val_sp:.4f}, {val_delta_r2:.4f}, {val_delta_pcc:.4f}, {val_delta_sp:.4f}")
            
            if avg_loss < loss:
                print(f'save model: epoch {epoch}')
                self.save_best_model = self.model
                loss = avg_loss
                torch.save(self.save_best_model.state_dict(), f"trans_models/seed{self.seed}/ca_model_lr{lr}.pth")

        if 'test_loader' in self.dataloader:
            test_loader = self.dataloader['test_loader']
            test_mse, test_r2, test_pcc, test_sp, test_delta_r2, test_delta_pcc, test_delta_sp \
                = self.evaluate_model(self.save_best_model, test_loader, self.device)
            print(f"Test matrix (MSE, R2, PCC, SP, delta_R2, delta_PCC, delta_SP): "
                  f"{test_mse:.4f}, {test_r2:.4f}, {test_pcc:.4f}, {test_sp:.4f}, {test_delta_r2:.4f}, {test_delta_pcc:.4f}, {test_delta_sp:.4f}")

    
    def evaluate_model(self, model, loader, device, no_perturb=False):
        model.eval()
        all_true, all_pred, all_ctrl = [], [], []
        drug_name, cell_type, smiles = [], [], []
        with torch.no_grad():
            for batch in loader:
                batch.to(device)
                data_ctrl = batch.x.to(device)
                data_pert = batch.y.to(device)
                mol_fp = batch.smiles_coding.to(device)
                cell_line = batch.cell_type_ids

                drug_name_i = batch.drug_name
                cell_type_i = batch.cell_type
                smiles_i = batch.smiles
                drug_name.extend(drug_name_i)
                cell_type.extend(cell_type_i)
                smiles.extend(smiles_i)
                
                y_true = data_pert.detach().cpu().numpy()
                x_ctrl = data_ctrl.detach().cpu().numpy()
                all_true.append(y_true)
                all_ctrl.append(x_ctrl)

                if no_perturb is False:
                    pert_pred, cond = model(data_ctrl, mol_fp, cell_line)
                    y_pred = pert_pred.detach().cpu().numpy()
                    all_pred.append(y_pred)

        all_true = np.concatenate(all_true, axis=0)
        all_ctrl = np.concatenate(all_ctrl, axis=0)

        if no_perturb is True:
            results = compute_metrics(all_true, all_ctrl)
            
            mse = results['MSE']
            r2 = results['R2']
            pcc = results['PCC']
            spearmanr = results['spearmanr']
            delta_r2 = 0
            delta_pcc = 0
            delta_spearmanr = 0
            
        else:  
            all_pred = np.concatenate(all_pred, axis=0)
            results = compute_metrics(all_true, all_pred)
            delta_pred = all_pred - all_ctrl
            delta_true = all_true - all_ctrl
            delta_result = compute_metrics(delta_true, delta_pred)

            # adata = self.write_h5ad(drug_name, cell_type, smiles, all_ctrl, all_pred, all_true)
            # adata.write('a375_mean_test_pred.h5ad')
        
            mse = results['MSE']
            r2 = results['R2']
            pcc = results['PCC']
            spearmanr = results['spearmanr']
            delta_r2 = delta_result['R2']
            delta_pcc = delta_result['PCC']
            delta_spearmanr = delta_result['spearmanr']

        return mse, r2, pcc, spearmanr, delta_r2, delta_pcc, delta_spearmanr

    def ctrl_evaluate(self):
        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader.get('val_loader')
        test_loader = self.dataloader['test_loader']

        tr_mse, tr_r2, tr_pcc, tr_sp, _, _, _ = self.evaluate_model(self.model, train_loader, self.device, no_perturb=True)
        print(f"Train matrix (MSE, R2, PCC): {tr_mse:.4f}, {tr_r2:.4f}, {tr_pcc:.4f}, {tr_sp:.4f}")

        val_mse, val_r2, val_pcc, val_sp, _, _, _ = self.evaluate_model(self.model, val_loader, self.device, no_perturb=True)
        print(f"Val matrix (MSE, R2, PCC): {val_mse:.4f}, {val_r2:.4f}, {val_pcc:.4f}, {val_sp:.4f}")

        test_mse, test_r2, test_pcc, test_sp, _, _, _ = self.evaluate_model(self.model, test_loader, self.device, no_perturb=True)
        print(f"Test matrix (MSE, R2, PCC): {test_mse:.4f}, {test_r2:.4f}, {test_pcc:.4f}, {test_sp:.4f}")

    def write_h5ad(self, drug_name, cell_type, smiles,
                all_ctrl=None, all_pred=None, all_true=None):

        obs = pd.DataFrame({
            "drug_name": list(drug_name),
            "cell_type": list(cell_type),
            "smiles": list(smiles)
        })

        var = pd.DataFrame(index=self.genes)
        var["gene_name"] = self.genes

        adata = ad.AnnData(
            X=all_ctrl.astype(np.float32),
            obs=obs,
            var=var
        )

        adata.layers["ctrl"] = all_ctrl.astype(np.float32)
        adata.layers["pred"] = all_pred.astype(np.float32)

        if all_true is not None:
            adata.layers["true"] = all_true.astype(np.float32)

        return adata

    def load_pretrained(self, path, strict=True):
        print(f"Loading pretrained model from: {path}")
        state_dict = torch.load(path, map_location=self.device)
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
            loader = self.dataloader['test_loader']
        else:
            loader = self.dataloader.get('val_loader')

        mse, r2, pcc, sp, delta_r2, delta_pcc, delta_sp = self.evaluate_model(model=self.best_model, loader=loader, device=self.device)
        print(f"Test matrix (MSE, R2, PCC, SP): {mse:.4f}, {r2:.4f}, {pcc:.4f}, {sp:.4f}")


    def predict(self, drug_names, smiles_list, cell_type, w_file, batch_size=32):
    
        ctrl_adata = self.ctrl_adata[self.ctrl_adata.obs["cell_type"] == cell_type]
        cell_type_id = torch.tensor(self.cell2id[cell_type], dtype=torch.long)
    
        dataset = DrugPredictDataset_transcript(
            drug_names, smiles_list, cell_type, cell_type_id, ctrl_adata, self.device)
    
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
    
        n_samples = len(dataset)
        n_genes = ctrl_adata.n_vars

        X_ctrl = np.zeros((n_samples, n_genes), dtype=np.float32)
        X_pred = np.zeros((n_samples, n_genes), dtype=np.float32)
    
        obs_dict = {
            "drug_name": np.empty(n_samples, dtype=object),
            "cell_type": np.empty(n_samples, dtype=object),
            "smiles": np.empty(n_samples, dtype=object)
        }
    
        model = self.best_model
        model.eval()
    
        offset = 0
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue

                batch = batch.to(self.device)
                ctrl_x = batch.x
                mol_fp = batch.smiles_coding
                cell_ids = batch.cell_type_ids

                pert_pred, _ = model(ctrl_x, mol_fp, cell_ids)
                bsz = pert_pred.shape[0]

                X_ctrl[offset:offset+bsz] = ctrl_x.cpu().numpy()
                X_pred[offset:offset+bsz] = pert_pred.cpu().numpy()

                start = offset
                end = offset + bsz
                obs_dict["drug_name"][start:end] = drug_names[start:end]
                obs_dict["cell_type"][start:end] = cell_type
                obs_dict["smiles"][start:end] = smiles_list[start:end]

                offset += bsz

                del batch, ctrl_x, mol_fp, cell_ids, pert_pred
                torch.cuda.empty_cache()
                gc.collect()    

        obs = pd.DataFrame({
            "drug_name": [str(x) for x in obs_dict["drug_name"]],
            "cell_type": [str(x) for x in obs_dict["cell_type"]],
            "smiles": [str(x) for x in obs_dict["smiles"]]
        })

        var = pd.DataFrame({"gene_name": self.genes}, index=self.genes)

        adata = ad.AnnData(
            X=X_ctrl.astype(np.float32),
            obs=obs,
            var=var
        )
        # adata.layers["ctrl"] = X_ctrl.astype(np.float32)
        adata.layers["pred"] = X_pred.astype(np.float32)

        adata.write_h5ad(w_file, compression="gzip")
        print(f"drug_predict_{cell_type}.h5ad saved (standard h5ad, max data supported)")
