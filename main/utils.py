import scanpy as sc
import warnings
import torch
import numpy as np
import pickle
import subprocess
from torch_geometric.data import Data
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint, DataStructs
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, average_precision_score


warnings.filterwarnings("ignore")
sc.settings.verbosity = 0


def calc_fingerprint(smiles, fp_type, radius=2, n_bits=1024):
    """
    Calculate a molecular fingerprint
    input: SMILES
    Output: NumPy vector
    """
    if not isinstance(smiles, str) or smiles.strip() == "":
        print('The smiles format is incorrect. Please check it carefully:')
        print(smiles)
        return np.zeros(n_bits, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles)
    if fp_type == "morgan":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    elif fp_type == "maccs":
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif fp_type == "rdkit":
        fp = RDKFingerprint(mol, fpSize=n_bits)
    else:
        raise ValueError(f"Unsupported fingerprint type: {fp_type}")
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def sanitize_smiles(smiles):
    if smiles is None:
        return None

    if isinstance(smiles, list):
        if len(smiles) == 0:
            return None
        smiles = smiles[0]

    smiles = str(smiles).strip()
    if smiles == "":
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    frags = Chem.GetMolFrags(mol, asMols=True)
    main_mol = max(frags, key=lambda m: m.GetNumAtoms())

    return Chem.MolToSmiles(main_mol, canonical=True)


def smiles_embedding(smiles, fp_type="rdkit", n_bits=1024):
    """
    Input: adata.obs['smiles'](pd.Series or list)
    Output: torch.Tensor  shape: [num_samples, n_bits]
    """

    smiles = sanitize_smiles(smiles)
    if smiles is None:
        return None

    fp = calc_fingerprint(smiles, fp_type=fp_type, n_bits=n_bits)
    return fp


def encode_cell_type(cell_type_series):
    """Encode cell line names to integer IDs"""
    unique_cells = sorted(cell_type_series.unique())
    cell2id = {c: i for i, c in enumerate(unique_cells)}
    ids = torch.tensor([cell2id[c] for c in cell_type_series], dtype=torch.long)
    return ids, cell2id


import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import spearmanr


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    逐样本计算 MSE, R², PCC, SpearmanR，然后取平均
    输入输出保持不变
    """

    assert y_true.shape == y_pred.shape, "y_true and y_pred shape must match."

    r2 = r2_score(y_true.reshape(-1), y_pred.reshape(-1))

    mask = ~(np.isnan(y_true).any(axis=1) | np.isnan(y_pred).any(axis=1))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n_samples = y_true.shape[0]
    if n_samples == 0:
        return {"MSE": np.nan, "R2": np.nan, "PCC": np.nan, "spearmanr": np.nan, "R2_single": np.nan}

    # MSE
    mse = np.mean((y_true - y_pred) ** 2, axis=1)
    mse_mean = mse.mean()

    # PCC
    yt_mean = y_true.mean(axis=1, keepdims=True)
    yp_mean = y_pred.mean(axis=1, keepdims=True)
    yt_center = y_true - yt_mean
    yp_center = y_pred - yp_mean
    numerator = np.sum(yt_center * yp_center, axis=1)
    denominator = np.sqrt(np.sum(yt_center**2, axis=1) * np.sum(yp_center**2, axis=1))

    with np.errstate(divide='ignore', invalid='ignore'):
        pcc = numerator / denominator
    pcc_mean = np.nanmean(pcc)

    # Spearman
    spear_list = []
    for yt, yp in zip(y_true, y_pred):
        try:
            spear_list.append(spearmanr(yt, yp)[0])
        except Exception:
            spear_list.append(np.nan)
    spear_mean = np.nanmean(spear_list)

    results = {
        "MSE": mse_mean,
        "R2": r2,
        "PCC": pcc_mean,
        "spearmanr": spear_mean
    }

    return results


def get_genes_from_perts(perts):
    if isinstance(perts, str):
        perts = [perts]

    genes = []
    for p in np.unique(perts):
        genes.extend(p.split('+'))

    genes = [g for g in genes if g != 'ctrl']
    return sorted(set(genes))


def compute_metrics_classify(y_true, y_pred=None, logits=None, threshold=0.5, return_curve_metrics=True):
    """
    Binary Classification Evaluation Function
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    y_true = y_true.flatten()

    # ===== logits -> probs -> pred =====
    probs = None
    if logits is not None:
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()

        logits = logits.reshape(-1)
        probs = 1 / (1 + np.exp(-logits))
        y_pred = (probs > threshold).astype(int)

    else:
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        y_pred = y_pred.flatten()

    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # ===== Precision / Recall / F1 =====
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics["precision"] = p
    metrics["recall"] = r
    metrics["f1"] = f1

    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    metrics["class_0_precision"] = p[0]
    metrics["class_0_recall"] = r[0]
    metrics["class_0_f1"] = f1[0]
    metrics["class_0_support"] = support[0]

    metrics["class_1_precision"] = p[1]
    metrics["class_1_recall"] = r[1]
    metrics["class_1_f1"] = f1[1]
    metrics["class_1_support"] = support[1]

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    # ===== AUC =====
    if probs is not None and return_curve_metrics:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, probs)
        except:
            metrics["roc_auc"] = None

        try:
            metrics["pr_auc"] = average_precision_score(y_true, probs)
        except:
            metrics["pr_auc"] = None

    return metrics


def parse_any_pert(pert):
    if 'ctrl' in pert and pert != 'ctrl':
        a, b = pert.split('+')
        return b if a == 'ctrl' else a
    elif 'ctrl' not in pert:
        return pert.split('+')
    return None


def create_cell_graph_dataset_for_prediction(
    drug_names, smiles_list, cell_type,
    smile_encode_type, cell_type_ids,
    ctrl_adata, device):

    X = ctrl_adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    ctrl_tensor = torch.tensor(X, dtype=torch.float32)

    graphs = []
    meta = []

    for i in range(len(smiles_list)):
        smiles_fp = smiles_embedding(smiles_list[i], fp_type="rdkit")
        if smiles_fp is None:
            continue

        smiles_fp = torch.tensor(smiles_fp, dtype=torch.float32)

        data = Data(
            x=ctrl_tensor.clone(),
            cell_type_ids=cell_type_ids.clone(),
            smiles_coding=smiles_fp.unsqueeze(0)
        )

        graphs.append(data.to(device))
        meta.append({
            "drug_name": drug_names[i],
            "smiles": smiles_list[i],
            "cell_type": cell_type
        })

    return graphs, meta

class DrugPredictDataset_transcript(Dataset):
    def __init__(self, drug_names, smiles_list, cell_type,
                 cell_type_id, ctrl_adata, device):
        self.drug_names = drug_names
        self.smiles_list = smiles_list
        self.cell_type = cell_type
        self.cell_type_id = cell_type_id
        self.device = device

        X = ctrl_adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()

        ctrl_mean = X.mean(axis=0)                     # (n_genes,)
        self.ctrl_tensor = torch.tensor(
            ctrl_mean, dtype=torch.float32
        )

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        smiles_fp = smiles_embedding(
            smiles, fp_type="rdkit"
        )
        if smiles_fp is None:
            return None

        smiles_fp = torch.tensor(smiles_fp, dtype=torch.float32)

        return Data(
            x=self.ctrl_tensor.unsqueeze(0),
            smiles_coding=smiles_fp.unsqueeze(0),
            cell_type_ids=self.cell_type_id.unsqueeze(0)
        )



class DrugPredictDataset_sensitive(Dataset):
    def __init__(self, drug_names, smiles_list, cell_type,
                 cell_type_id, ctrl_adata, device):

        self.cell_type_id = cell_type_id
        self.device = device

        # Filter Out Invalid SMILES
        self.data = []
        for d, s in zip(drug_names, smiles_list):
            fp = smiles_embedding(s, fp_type="rdkit")
            if fp is not None:
                self.data.append((d, s, fp))

        self.meta = [(d, s) for d, s, _ in self.data]

        X = ctrl_adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()

        ctrl_mean = X.mean(axis=0)
        self.ctrl_tensor = torch.tensor(ctrl_mean, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        drug_name, smiles, smiles_fp = self.data[idx]

        smiles_fp = torch.tensor(smiles_fp, dtype=torch.float32)

        return Data(
            x=self.ctrl_tensor.unsqueeze(0),
            smiles_coding=smiles_fp.unsqueeze(0),
            cell_type_ids=self.cell_type_id.unsqueeze(0)
        )


class DataSplitter:
    def __init__(self, adata, split_type='random', seen=0):
        self.adata = adata
        self.split_type = split_type
        self.seen = seen

    def split_data(self, test_size=0.1, test_perts=None, split_name='split',
                   seed=None, val_size=0.1):
        np.random.seed(seed)
        if self.split_type in ['random', 'random_no_test']:
            unique_perts = [p for p in self.adata.obs['condition'].unique() if p.split('_')[1] != 'none']
        elif self.split_type in ['drug', 'drug_no_test']:
            unique_perts = [p for p in self.adata.obs['drug_name'].unique() if p != 'none']

        if self.split_type.endswith('no_test'):
            train, val = self.get_split_list(unique_perts, test_size=val_size)
            test = []
        else:
            train, test = self.get_split_list(unique_perts, test_size=test_size, test_perts=test_perts)
            train, val = self.get_split_list(train, test_size=val_size)

        mapping = {x: 'train' for x in train}
        mapping.update({x: 'val' for x in val})
        mapping.update({x: 'test' for x in test})
        # mapping.update({'ctrl': 'train'})
        if self.split_type in ['random', 'random_no_test']:
            self.adata.obs[split_name] = self.adata.obs['condition'].map(mapping)
        elif self.split_type in ['drug', 'drug_no_test']:
            self.adata.obs[split_name] = self.adata.obs['drug_name'].map(mapping)
        return self.adata

    def get_split_list(self, pert_list, test_size=0.1, test_perts=None):
        unique_perts = np.unique(pert_list)

        if test_perts == None:
            test_perts = np.random.choice(unique_perts, int(len(unique_perts) * test_size))
        else:
            test_perts = test_perts

        train_perts = [p for p in pert_list if p not in test_perts]
        return train_perts, list(test_perts)

