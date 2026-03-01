# src/model_trainer.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelTrainer:
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            print(f"\n{name}")
            print("Accuracy:", accuracy_score(y_test, predictions))
            print("Precision:", precision_score(y_test, predictions))
            print("Recall:", recall_score(y_test, predictions))
            print("F1 Score:", f1_score(y_test, predictions))
