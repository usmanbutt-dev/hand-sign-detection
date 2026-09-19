"""Quick runner to prepare the keypoints CSV from downloaded datasets."""
import sys
sys.path.insert(0, ".")

from src.data.prepare import prepare_keypoints_csv

df = prepare_keypoints_csv(
    npy_dir="data/processed/asl_landmarks/landmarks",
    csv_files=None,
    output_path="data/processed/keypoints.csv",
)

print(f"\nFINAL SHAPE: {df.shape}")
print(f"CLASSES: {sorted(df['class'].unique())}")
print(f"SAMPLES PER CLASS:\n{df['class'].value_counts().sort_index().to_string()}")
