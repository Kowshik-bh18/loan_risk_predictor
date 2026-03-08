from pathlib import Path
from data_loader import DataLoader
from pipeline_trainer import PipelineTrainer
import os

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = os.path.join(BASE_DIR,"data","loan_data.csv")

    loader = DataLoader(DATA_PATH)
    data = loader.load_data()

    trainer = PipelineTrainer()
    trainer.train(data)


if __name__ == "__main__":
    main()