from pathlib import Path
from pathlib import Path
from data_loader import DataLoader
from preprocessor import Preprocessor
from model_trainer import ModelTrainer
import os
def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = os.path.join(BASE_DIR,"data","loan_data.csv")

    # Load data
    loader = DataLoader(DATA_PATH)
    data = loader.load_data()
    print(data["Loan_Status"].value_counts())
    # Preprocess data
    preprocessor = Preprocessor(data)
    X_train, X_test, y_train, y_test = preprocessor.process()

    # Train models
    trainer = ModelTrainer()
    results = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
    

    
    


