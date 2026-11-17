import numpy as np
import pandas as pd

X = np.load("shadow_attack_features.npy")
y = np.load("shadow_attack_labels.npy")

print("Features shape:", X.shape)
print("Labels shape:", y.shape)

df = pd.DataFrame(X, columns=["prob", "dist_from_boundary", "recon_error", "latent_diff_norm"])
df["member"] = y.astype(int)

print(df.head(10))
print(df.describe())
print(df.groupby("member").mean())