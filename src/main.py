from pathlib import Path
from data_loader import DataLoader
from preprocessor import Preprocessor
import os
def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = os.path.join(BASE_DIR,"data","loan_data.csv")

    loader = DataLoader(DATA_PATH)
    data = loader.load_data()

    preprocessor = Preprocessor(data)
    X_train, X_test, y_train, y_test = preprocessor.process()

    print("\nShape of Training Data:", X_train.shape)
    print("Shape of Test Data:", X_test.shape)

if __name__ == "__main__":
    main()
    


