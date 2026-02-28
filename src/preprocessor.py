# src/preprocessor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.X = None
        self.y = None

    def encode_target(self):
        """Convert Yes/No to 1/0"""
        self.data['loan_approved'] = self.data['loan_approved'].map({
            'Yes': 1,
            'No': 0
        })

    def encode_categorical(self):
        """One-hot encode employment_status"""
        self.data = pd.get_dummies(self.data, columns=['employment_status'], drop_first=True)

    def separate_features_target(self):
        """Split features and target"""
        self.X = self.data.drop('loan_approved', axis=1)
        self.y = self.data['loan_approved']

    def convert_to_numpy(self):
        """Convert to NumPy arrays"""
        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def split_data(self, test_size=0.2, random_state=42):
        """Train-test split"""
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def process(self):
        """Complete preprocessing pipeline"""
        self.encode_target()
        self.encode_categorical()
        self.separate_features_target()
        self.convert_to_numpy()
        return self.split_data()