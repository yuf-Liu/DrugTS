import pandas as pd
import numpy as np
import anndata as ad
import warnings
import sys

warnings.filterwarnings("ignore")

# ============================
# file
# ============================
ic50_file = "ic50_labels_numeric_filtered.csv"
drug_file = "DRUG_MASTER_5DB_MERGED.csv"
exp_file = "CCLE_EXP.csv"
genes_file = "genes.txt"
out_file = "processed_data.h5ad"

print("Read IC50 data...")
ic50 = pd.read_csv(ic50_file, index_col=0, dtype=str)
ic50 = ic50.apply(pd.to_numeric, errors='coerce')

print("Read Drug data...")
drug_master = pd.read_csv(drug_file, dtype=str).dropna(subset=['pubchem_cid'])
drug_master = drug_master.drop_duplicates(subset='pubchem_cid', keep='first')
drug_info = drug_master.set_index('pubchem_cid')[['primary_name', 'smiles_canon']]
drug_info.index = drug_info.index.astype(str)

print("Read RNA-seq data...")
exp = pd.read_csv(exp_file, index_col=0)

print("Read target genes...")
genes_list = pd.read_csv(genes_file, header=None)[0].tolist()

existing_genes = [g for g in genes_list if g in exp.columns]
missing_genes = [g for g in genes_list if g not in exp.columns]

if missing_genes:
    print(f"Missing gene (removed, total {len(missing_genes)} ):", len(missing_genes))

exp = exp[existing_genes]

exp_np = exp.to_numpy()
cell_to_idx = {c: i for i, c in enumerate(exp.index)}

print("Building AnnData Content (Reading IC50 Values from the Original Table)...")

obs_list = []
X_list = []

for cell in ic50.index:

    for drug in ic50.columns:
        ic50_value = ic50.at[cell, drug]
        if pd.isna(ic50_value):
            continue  # 空值跳过

        if drug in drug_info.index:
            primary_name = drug_info.at[drug, "primary_name"]
            smiles = drug_info.at[drug, "smiles_canon"]
        else:
            primary_name = drug
            smiles = ""

        obs_list.append({
            "cell_type": cell,
            "drug_name": drug,
            "drug_name_primary": primary_name,
            "smiles": smiles,
            "condition": f"{cell}_{drug}",
            "ic50": ic50_value
        })

        X_list.append(exp_np[cell_to_idx[cell]])

X = np.array(X_list, dtype=np.float32)
obs = pd.DataFrame(obs_list)
var = pd.DataFrame({"gene": existing_genes})

dup_mask = obs.duplicated(subset=["cell_type", "drug_name"], keep=False)
if dup_mask.any():
    print("Warning: The processed data contains duplicate (cell-drug) pairs!")
    print(obs.loc[dup_mask, ["cell_type", "drug_name", "ic50"]])
else:
    print("The processed data contains no duplicate (cell, drug) pairs; each pair is unique.")

adata = ad.AnnData(X=X, obs=obs, var=var)
adata.write(out_file)
print("Done! The file has been saved to:", out_file)
