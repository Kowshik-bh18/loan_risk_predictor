import joblib
import pandas as pd


class Predictor:

    def __init__(self, model_path="model/pipeline.pkl"):
        self.pipeline = joblib.load(model_path)

    def predict(self, input_dict):

        input_df = pd.DataFrame([input_dict])

        prediction = self.pipeline.predict(input_df)[0]

        probability = self.pipeline.predict_proba(input_df)[0][1]

        result = "Loan Approved" if prediction == 1 else "Loan Rejected"

        return result, round(probability * 100, 2)