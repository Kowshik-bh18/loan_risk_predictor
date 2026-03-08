import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class PipelineTrainer:

    def __init__(self):

        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }

    def train(self, data):

        # Drop ID column
        if "Loan_ID" in data.columns:
            data = data.drop("Loan_ID", axis=1)

        # Handle missing values
        for col in data.columns:
            if data[col].dtype == "object":
                data[col] = data[col].fillna(data[col].mode()[0])
            else:
                data[col] = data[col].fillna(data[col].median())

        # Encode target
        data["Loan_Status"] = data["Loan_Status"].map({"Y": 1, "N": 0})

        X = data.drop("Loan_Status", axis=1)
        y = data["Loan_Status"]

        # Identify column types
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = X.select_dtypes(include=["object"]).columns

        # Preprocessing pipelines
        numeric_pipeline = Pipeline([
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ])

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        best_model = None
        best_score = 0
        best_name = ""

        for name, model in self.models.items():

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)

            preds = pipeline.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)

            print(f"\n{name}")
            print("Accuracy:", acc)
            print("Precision:", prec)
            print("Recall:", rec)
            print("F1 Score:", f1)

            if f1 > best_score:
                best_score = f1
                best_model = pipeline
                best_name = name

        os.makedirs("model", exist_ok=True)

        joblib.dump(best_model, "model/pipeline.pkl")

        print("\nBest Model:", best_name)
        print("Best F1 Score:", best_score)
        print("Pipeline saved to model/pipeline.pkl")