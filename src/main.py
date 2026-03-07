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
    X_train, X_test, y_train, y_test, feature_names = preprocessor.process()

    # Train models
    trainer = ModelTrainer()
    results = trainer.train_and_evaluate(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names
)
    
from predictor import Predictor

predictor = Predictor()

sample_input = [5000, 2000, 150, 360, 1, 0, 1, 0]

result = predictor.predict(sample_input)

print("\nPrediction:", result)

if __name__ == "__main__":
    main()
    

    
    


