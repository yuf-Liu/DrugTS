import pandas as pd
import numpy as np
import os

input_file = "ic50_labels_numeric_filtered.csv"

output_dir = "downsample_results"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_file, index_col=0)

# random seed
seeds = [42, 123, 2024]

for run_id, seed in enumerate(seeds, start=1):
    print(f"\n===== Run {run_id} | seed={seed} =====")

    np.random.seed(seed)

    df_sampled = df.copy()
    stats_lines = []

    for cell_line in df.index:
        row = df.loc[cell_line]

        idx_0 = row[row == 0].index
        idx_1 = row[row == 1].index

        n0, n1 = len(idx_0), len(idx_1)

        target_n2 = n0 + n1

        if n0 > n1:
            sampled_idx_2 = np.random.choice(idx_0, size=n1, replace=False)
        else:
            sampled_idx_2 = idx_0

        drop_idx_2 = set(idx_0) - set(sampled_idx_2)
        df_sampled.loc[cell_line, list(drop_idx_2)] = np.nan

        new_row = df_sampled.loc[cell_line]
        new_n0 = (new_row == 0).sum()
        new_n1 = (new_row == 1).sum()

        stats_lines.append(
            f"{cell_line}\n"
            f"Before: 0={n0}, 1={n1}\n"
            f"After : 0={new_n0}, 1={new_n1}\n"
            f"{'-'*40}\n"
        )

    csv_path = os.path.join(output_dir, f"downsampled_run{run_id}.csv")
    txt_path = os.path.join(output_dir, f"stats_run{run_id}.txt")

    df_sampled.to_csv(csv_path)

    with open(txt_path, "w") as f:
        f.writelines(stats_lines)

    print(f"Saved: {csv_path}")
    print(f"Saved: {txt_path}")

print("\nAll finish！")
