"""
Model training and evaluation utilities for Human Activity Recognition
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import numpy as np
import joblib
import os


class HARModelTrainer:
    """Handler for training and evaluating ML models"""

    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.probabilities = {}
        self.metrics = {}
        self.confusion_matrices = {}

    def train_knn(self, X_train, y_train, n_neighbors=5):
        """Train K-Nearest Neighbors classifier"""
        y_train_flat = y_train.values.ravel() if hasattr(y_train, 'values') else y_train

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train_flat)
        self.models['KNN'] = knn
        return knn

    def train_svm(self, X_train, y_train, kernel='rbf', random_state=42):
        """Train Support Vector Machine classifier"""
        y_train_flat = y_train.values.ravel() if hasattr(y_train, 'values') else y_train

        svm = SVC(kernel=kernel, random_state=random_state, probability=True)
        svm.fit(X_train, y_train_flat)
        self.models['SVM'] = svm
        return svm

    def train_logistic_regression(self, X_train, y_train, max_iter=1000, random_state=42):
        """Train Logistic Regression classifier"""
        y_train_flat = y_train.values.ravel() if hasattr(y_train, 'values') else y_train

        log_reg = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state
        )
        log_reg.fit(X_train, y_train_flat)
        self.models['Logistic Regression'] = log_reg
        return log_reg

    def train_random_forest(self, X_train, y_train, n_estimators=100, random_state=42):
        """Train Random Forest classifier"""
        y_train_flat = y_train.values.ravel() if hasattr(y_train, 'values') else y_train

        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        rf.fit(X_train, y_train_flat)
        self.models['Random Forest'] = rf
        return rf

    def train_decision_tree(self, X_train, y_train, random_state=42):
        """Train Decision Tree classifier"""
        y_train_flat = y_train.values.ravel() if hasattr(y_train, 'values') else y_train

        dt = DecisionTreeClassifier(random_state=random_state)
        dt.fit(X_train, y_train_flat)
        self.models['Decision Tree'] = dt
        return dt

    def train_all_models(self, X_train, y_train):
        """Train all classifiers"""
        print("Training KNN...")
        self.train_knn(X_train, y_train)

        print("Training SVM...")
        self.train_svm(X_train, y_train)

        print("Training Logistic Regression...")
        self.train_logistic_regression(X_train, y_train)

        print("Training Random Forest...")
        self.train_random_forest(X_train, y_train)

        print("Training Decision Tree...")
        self.train_decision_tree(X_train, y_train)

        print("All models trained successfully!")
        return self.models

    def predict(self, X, model_name=None):
        """Make predictions with specified model or all models"""
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not trained yet")

            model = self.models[model_name]
            predictions = model.predict(X)
            probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None

            return predictions, probabilities
        else:
            # Predict with all models
            for name, model in self.models.items():
                self.predictions[name] = model.predict(X)
                if hasattr(model, 'predict_proba'):
                    self.probabilities[name] = model.predict_proba(X)

            return self.predictions, self.probabilities

    def evaluate(self, X, y_true, model_name=None):
        """Evaluate model(s) on given data"""
        y_true_flat = y_true.values.ravel() if hasattr(y_true, 'values') else y_true

        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not trained yet")

            y_pred = self.models[model_name].predict(X)
            return self._calculate_metrics(y_true_flat, y_pred, model_name)
        else:
            # Evaluate all models
            for name, model in self.models.items():
                y_pred = model.predict(X)
                self._calculate_metrics(y_true_flat, y_pred, name)

            return self.metrics

    def _calculate_metrics(self, y_true, y_pred, model_name):
        """Calculate evaluation metrics"""
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.confusion_matrices[model_name] = cm.tolist()

        self.metrics[model_name] = metrics
        return metrics

    def get_best_model(self):
        """Get the best performing model based on F1 score"""
        if not self.metrics:
            raise ValueError("No models evaluated yet")

        best_model_name = max(self.metrics, key=lambda x: self.metrics[x]['f1_score'])
        return best_model_name, self.models[best_model_name], self.metrics[best_model_name]

    def save_model(self, model_name, filepath):
        """Save a trained model to disk"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.models[model_name], filepath)
        print(f"Model {model_name} saved to {filepath}")

    def load_model(self, model_name, filepath):
        """Load a trained model from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        self.models[model_name] = joblib.load(filepath)
        print(f"Model {model_name} loaded from {filepath}")
        return self.models[model_name]

    def save_all_models(self, directory='backend/models/saved_models'):
        """Save all trained models"""
        for model_name in self.models:
            filename = model_name.lower().replace(' ', '_') + '.pkl'
            filepath = os.path.join(directory, filename)
            self.save_model(model_name, filepath)

    def get_metrics_summary(self):
        """Get summary of all model metrics"""
        return {
            'metrics': self.metrics,
            'confusion_matrices': self.confusion_matrices,
            'model_count': len(self.models)
        }
