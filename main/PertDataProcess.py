import os
import pickle
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
import scanpy as sc
from tqdm import tqdm
from utils import DataSplitter, smiles_embedding, encode_cell_type

import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0


class PertData:
    def __init__(self, data_path):
        self.data_path = data_path

    def load(self, data_name=None, test_file=None):
        """Load the h5ad data and build/read the cell graph dataset"""
        if not os.path.exists(self.data_path):
            raise ValueError("h5ad file path does not exist")

        data_path = os.path.join(self.data_path, data_name)
        adata_file = test_file or 'perturb_processed.h5ad'
        self.adata = sc.read_h5ad(os.path.join(data_path, adata_file))
        self.dataset_name = os.path.basename(data_path)
        self.dataset_path = data_path

        self.gene_names = self.adata.var.gene.tolist()
        self.ctrl_adata = self.adata[self.adata.obs['drug_name'] == 'none']
        self.cell_type = np.unique(self.adata.obs['cell_type'].tolist())   
        cell_ids, self.cell2id = encode_cell_type(self.adata.obs['cell_type'])    

    def prepare_split(self, task='transcriptome', split='random', seed=1, train_gene_set_size=0.8, test_gene_set_size=0.1):
        """train/val/test split"""
        valid_splits = ['random', 'random_no_test', 'drug', 'drug_no_test']
        if split not in valid_splits:
            raise ValueError(f"Invalid split, must be one of {valid_splits}")
        
        dataset_fname = os.path.join(self.dataset_path, 'data_pyg', split + '_' + '_cell_graphs.pkl')
        os.makedirs(os.path.dirname(dataset_fname), exist_ok=True)

        if os.path.isfile(dataset_fname):
            print("Loading cached pyg dataset...")
            self.dataset_processed = pickle.load(open(dataset_fname, "rb"))
        else:
            print("Creating pyg dataset...")
            if split in ['random', 'random_no_test']:
                print('random split:')
                group_col = 'condition'
            elif split in ['drug', 'drug_no_test']:
                print('Drug-based split:')
                group_col = 'drug_name'
            else:
                raise ValueError(f"Don't support : {split}")

            if task == 'transcriptome':
                process_func = self.create_cell_graph_dataset_transcript
            else:
                process_func = self.create_cell_graph_dataset_sensitive

            unique_keys = self.adata.obs[group_col].unique()
            self.dataset_processed = {
                key: process_func(self.adata, key, split)
                for key in tqdm(unique_keys, desc=f"Processing {task}")
            }

            pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))
            print("Saved dataset to", dataset_fname)

        self.split, self.seed = split, seed
        split_folder = os.path.join(self.dataset_path, 'splits')
        os.makedirs(split_folder, exist_ok=True)
        split_file = f"{self.dataset_name}_{split}_{seed}_{train_gene_set_size}.pkl"
        split_path = os.path.join(split_folder, split_file)

        if os.path.exists(split_path):
            print("Loading cached split...")
            self.set2conditions = pickle.load(open(split_path, "rb"))
            return

        print("Creating new split...")
        if split in ['random', 'random_no_test']:
            DS = DataSplitter(self.adata, split_type=split)
            adata = DS.split_data(test_size=test_gene_set_size, seed=seed)
            set2conditions = dict(adata.obs.groupby('split')['condition'].unique())
        elif split in ['drug', 'drug_no_test']:
            DS = DataSplitter(self.adata, split_type=split)
            adata = DS.split_data(test_size=test_gene_set_size, seed=seed)
            set2conditions = dict(adata.obs.groupby('split')['drug_name'].unique())
        else:
            raise ValueError(f"Unsupported split: {split}")

        self.set2conditions = {k: v.tolist() for k, v in set2conditions.items()}
        pickle.dump(self.set2conditions, open(split_path, "wb"))
        print("Saved split to", split_path)

    def get_dataloader(self, batch_size, test_batch_size=None):
        """return DataLoader"""
        test_batch_size = test_batch_size or batch_size
        cell_graphs = {}

        def collect_graphs(split_name):
            graphs = []
            for p in self.set2conditions.get(split_name, []):
                if p == 'none':
                    continue
                g_list = self.dataset_processed.get(p)
                if g_list is None:
                    print(f"[Warning] Skipping '{p}' — dataset_processed[{p}] is None")
                    continue
                graphs.extend(g_list)
            return graphs

        splits = ['train', 'val'] if self.split.endswith('no_test') else ['train', 'val', 'test']
        for s in splits:
            cell_graphs[s] = collect_graphs(s)

        self.dataloader = {
            'train_loader': DataLoader(cell_graphs['train'], batch_size=batch_size, shuffle=True, drop_last=True),
            'val_loader': DataLoader(cell_graphs['val'], batch_size=batch_size, shuffle=True)
        }
        if 'test' in cell_graphs:
            self.dataloader['test_loader'] = DataLoader(cell_graphs['test'], batch_size=test_batch_size, shuffle=False)
        return self.dataloader

    def create_cell_graph_dataset_transcript(self, adata, pert_category, split):
        """Build a cell graph dataset for a certain perturbation"""

        if 'none' in pert_category:
            return []

        adata_ctrl = adata[adata.obs['drug_name'] == 'none']
        ctrl_by_cell = {
            ct: adata_ctrl[adata_ctrl.obs['cell_type'] == ct]
            for ct in adata_ctrl.obs['cell_type'].unique()
        }

        if not hasattr(self, "_smiles_cache"):
            self._smiles_cache = {}

        if split in ['random', 'random_no_test']:
            adata_ = adata[adata.obs['condition'] == pert_category]
            obs = adata_.obs.reset_index(drop=True)
            X_pert = adata_.X
            if hasattr(X_pert, "toarray"):
                X_pert = X_pert.toarray()
            cell_graph = []
            for i in range(len(adata_)):
                row = obs.iloc[i]
                cell_type = row['cell_type']
                drug_name = row['drug_name']
                smiles = row['smiles']

                ctrl_adata = ctrl_by_cell.get(cell_type, None)
                if ctrl_adata is None or len(ctrl_adata) == 0:
                    print(f"Absence of {cell_type} unperturbed transcriptome")
                    continue

                X_ctrl = ctrl_adata.X
                if hasattr(X_ctrl, "toarray"):
                    X_ctrl = X_ctrl.toarray()

                cell_type_ids = torch.tensor(
                    [self.cell2id[cell_type]],
                    dtype=torch.long
                )

                if smiles in self._smiles_cache:
                    smiles_coding = self._smiles_cache[smiles]
                else:
                    smiles_coding = torch.tensor(
                        smiles_embedding(
                            smiles,
                            fp_type="rdkit"
                        )
                    ).unsqueeze(0)
                    self._smiles_cache[smiles] = smiles_coding
                self.smiles_coding_dim = smiles_coding.shape[-1]

                cell_graph.append(
                    self.create_cell_graph(X_ctrl, X_pert[i:i + 1], pert_category, [cell_type], cell_type_ids, drug_name, smiles, smiles_coding)
                )
            return cell_graph

        elif split in ['drug', 'drug_no_test']:
            adata_ = adata[adata.obs['drug_name'] == pert_category]
            obs = adata_.obs.reset_index(drop=True)
            X_pert = adata_.X
            if hasattr(X_pert, "toarray"):
                X_pert = X_pert.toarray()
            cell_graph = []
            for i in range(len(adata_)):
                row = obs.iloc[i]
                cell_type = row['cell_type']
                pert_i = row['condition']
                smiles_i = row['smiles']

                ctrl_adata = ctrl_by_cell.get(cell_type, None)
                if ctrl_adata is None or len(ctrl_adata) == 0:
                    print(f"Absence of {cell_type} unperturbed transcriptome")
                    continue
                X_ctrl = ctrl_adata.X
                if hasattr(X_ctrl, "toarray"):
                    X_ctrl = X_ctrl.toarray()
                X_pert_i = X_pert[i]

                cell_type_ids = torch.tensor(
                    [self.cell2id[cell_type]],
                    dtype=torch.long
                )

                if smiles_i in self._smiles_cache:
                    smiles_coding = self._smiles_cache[smiles_i]
                else:
                    smiles_coding = torch.tensor(
                        smiles_embedding(
                            smiles_i,
                            fp_type="rdkit"
                        )
                    ).unsqueeze(0)
                    self._smiles_cache[smiles_i] = smiles_coding
                self.smiles_coding_dim = smiles_coding.shape[-1]

                cell_graph.append(
                    self.create_cell_graph(X_ctrl, X_pert_i, pert_i, cell_type, cell_type_ids, pert_category, smiles_i, smiles_coding)
                )
            return cell_graph if len(cell_graph) > 0 else []


    @staticmethod
    def create_cell_graph_trans(X, y, pert_category, cell_type, cell_type_ids, drug_name, smiles, smiles_coding):
        if isinstance(pert_category, list):
            pert_category = pert_category[0]
        if isinstance(cell_type, list):
            cell_type = cell_type[0]
        if isinstance(drug_name, list):
            drug_name = drug_name[0]
        if isinstance(smiles, list):
            smiles = smiles[0]

        return Data(
            x=torch.tensor(X, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.float32),
            pert_category=pert_category,
            cell_type=cell_type,
            cell_type_ids=cell_type_ids,
            drug_name=drug_name,
            smiles=smiles,
            smiles_coding=smiles_coding
        )

    def create_cell_graph_dataset_sensitive(self, adata, pert_category, split):
        """Build a cell graph dataset for a certain perturbation"""

        if 'none' in pert_category:
            return []

        if split in ['random', 'random_no_test']:
            adata_sub = adata[adata.obs['condition'] == pert_category]
        elif split in ['drug', 'drug_no_test']:
            adata_sub = adata[adata.obs['drug_name'] == pert_category]
        else:
            return []

        if len(adata_sub) == 0:
            return []

        smiles_list = adata_sub.obs['smiles'].unique()
        smiles_cache = {}
        for s in smiles_list:
            emb = smiles_embedding([s], fp_type="rdkit")
            if emb is not None:
                smiles_cache[s] = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
            else:
                print(f"Smiles error, skip: {s}")

        cell_graph = []
        for i in range(len(adata_sub)):
            obs = adata_sub[i].obs

            if split in ['random', 'random_no_test']:
                X_i = adata_sub[i].X
                if hasattr(X_i, "toarray"):
                    X_i = X_i.toarray()
                pert_i = obs['condition'].iloc[0]
                drug_i = obs['drug_name'].iloc[0]
                cell_type_i = obs['cell_type'].iloc[0]
                ic50_i = obs['ic50'].iloc[0]
                smiles_i = obs['smiles'].iloc[0]
                cell_type_ids = torch.tensor([self.cell2id[cell_type_i]], dtype=torch.long)

            elif split in ['drug', 'drug_no_test']:
                pert_i = obs['condition'].iloc[0]
                adata_i = adata_sub[adata_sub.obs['condition'] == pert_i]
                X_i = adata_i.X
                if hasattr(X_i, "toarray"):
                    X_i = X_i.toarray()
                cell_type_i = adata_i.obs['cell_type'].tolist()[0]
                ic50_i = adata_i.obs['ic50'].tolist()[0]
                smiles_i = adata_i.obs['smiles'].tolist()[0]
                drug_i = pert_category
                cell_type_ids = torch.tensor([self.cell2id[cell_type_i]], dtype=torch.long)

            if smiles_i not in smiles_cache:
                continue
            smiles_coding = smiles_cache[smiles_i]
            self.smiles_coding_dim = len(smiles_coding)

            cell_graph.append(
                self.create_cell_graph(X_i, pert_i, ic50_i, cell_type_i, cell_type_ids, drug_i, smiles_i, smiles_coding)
            )

        return cell_graph if len(cell_graph) > 0 else []

    @staticmethod
    def create_cell_graph_sensitive(X, pert_category, ic50, cell_type, cell_type_ids, drug_name, smiles, smiles_coding):
        """Build a graph of a cell"""
        if isinstance(pert_category, list):
            pert_category = pert_category[0]
        if isinstance(cell_type, list):
            cell_type = cell_type[0]
        if isinstance(drug_name, list):
            drug_name = drug_name[0]
        if isinstance(smiles, list):
            smiles = smiles[0]

        return Data(
            x=torch.Tensor(X),
            pert_category=pert_category,
            ic50=ic50,
            cell_type=cell_type,
            cell_type_ids=cell_type_ids,
            drug_name=drug_name,
            smiles=smiles,
            smiles_coding=smiles_coding
        )