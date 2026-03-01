# src/preprocessor.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
class Preprocessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def clean_data(self):
        # Drop ID column
        if "Loan_ID" in self.data.columns:
            self.data = self.data.drop("Loan_ID", axis=1)

        # Fill missing values
        for column in self.data.columns:
            if self.data[column].dtype == "object":
                self.data[column] = self.data[column].fillna(self.data[column].mode()[0])
            else:
                self.data[column] = self.data[column].fillna(self.data[column].median())

    def encode_target(self):
        self.data["Loan_Status"] = self.data["Loan_Status"].map({
            "Y": 1,
            "N": 0
        })

    def encode_categorical(self):
        categorical_cols = self.data.select_dtypes(include=["object"]).columns
        self.data = pd.get_dummies(self.data, columns=categorical_cols, drop_first=True)

    def separate_features_target(self):
        X = self.data.drop("Loan_Status", axis=1)
        y = self.data["Loan_Status"]
        return X, y

    def process(self):
        self.clean_data()
        self.encode_target()
        self.encode_categorical()

        X, y = self.separate_features_target()

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)