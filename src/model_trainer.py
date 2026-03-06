# src/model_trainer.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd


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

        results = {}

        for name, model in self.models.items():

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            # Metrics
            train_accuracy = accuracy_score(y_train, train_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            precision = precision_score(y_test, test_pred)
            recall = recall_score(y_test, test_pred)
            f1 = f1_score(y_test, test_pred)

            # Store results
            results[name] = {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

            # Print metrics
            print(f"\n{name}")
            print("Train Accuracy:", train_accuracy)
            print("Test Accuracy:", test_accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1 Score:", f1)

            # Feature importance for tree models
            if hasattr(model, "feature_importances_"):

                importance = model.feature_importances_

                feature_importance = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importance
                }).sort_values(by="Importance", ascending=False)

                print("\nTop Important Features:")
                print(feature_importance.head(5))

        return results