import joblib
import numpy as np


class Predictor:

    def __init__(self, model_path="model/best_model.pkl"):
        self.model = joblib.load(model_path)

    def predict(self, input_data):

        # Convert input to NumPy array
        input_array = np.array(input_data).reshape(1, -1)

        prediction = self.model.predict(input_array)

        if prediction[0] == 1:
            return "Loan Approved"
        else:
            return "Loan Rejected"