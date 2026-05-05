from __future__ import annotations

from ml_pipeline import train_and_save


if __name__ == "__main__":
    bundle = train_and_save()
    print("Training complete.")
    print(f"Dataset source: {bundle['dataset_source']}")
    for name, metric in bundle["metrics"].items():
        print(f"{name}: {metric['accuracy']}% accuracy")
