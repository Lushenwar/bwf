import logging
from src.train import train_and_evaluate

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("🚀 Starting Shuttle-X Model Training...")
    train_and_evaluate()
    print("✅ Training Complete! Models saved to models/")
