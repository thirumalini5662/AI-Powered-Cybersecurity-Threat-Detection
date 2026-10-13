from src.generate_data import generate_dataset
from src.train_model import train_models
from src.detect_threats import run_detection
from src.visualize import create_visualizations

if __name__ == "__main__":
    generate_dataset()
    train_models()
    run_detection()
    create_visualizations()
    print("\nPipeline completed successfully.")
