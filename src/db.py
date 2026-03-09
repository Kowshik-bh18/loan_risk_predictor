from pymongo import MongoClient
from datetime import datetime


class MongoLogger:

    def __init__(self):

        uri = "mongodb+srv://vk:1817@cluster0.azn0yvf.mongodb.net/?appName=Cluster0"

        self.client = MongoClient(uri)

        self.db = self.client["loan_risk_db"]

        self.collection = self.db["predictions"]

    def log_prediction(self, data, result, probability):

        record = {
            "input": data,
            "prediction": result,
            "confidence": probability,
            "timestamp": datetime.now()
        }

        self.collection.insert_one(record)