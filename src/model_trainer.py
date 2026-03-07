# src/model_trainer.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os


class ModelTrainer:

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, feature_names):

        best_model = None
        best_f1 = 0
        best_model_name = ""

        results = {}

        for name, model in self.models.items():

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)

            results[name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

            print(f"\n{name}")
            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1 Score:", f1)

            # Track best model
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_model_name = name

        # Save best model
        os.makedirs("model", exist_ok=True)

        joblib.dump(best_model, "model/best_model.pkl")

        print("\nBest Model Selected:", best_model_name)
        print("Best F1 Score:", best_f1)
        print("Model saved to: model/best_model.pkl")

        return results