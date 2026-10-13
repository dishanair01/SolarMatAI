from matminer.datasets import load_dataset

print("Loading the materials dataset...")

df = load_dataset("matbench_mp_gap")

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())