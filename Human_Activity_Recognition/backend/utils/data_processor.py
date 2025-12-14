"""
Data processing utilities for Human Activity Recognition
"""
import pandas as pd
import numpy as np
import collections
import os
from sklearn.model_selection import train_test_split


class HARDataProcessor:
    """Handler for loading and preprocessing HAR dataset"""

    def __init__(self, dataset_path='UCI HAR Dataset/'):
        self.dataset_path = dataset_path
        self.activity_labels = None
        self.features = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_new = None
        self.X_val = None
        self.y_train_new = None
        self.y_val = None

    def check_dataset_exists(self):
        """Check if dataset is downloaded"""
        return os.path.exists(self.dataset_path)

    def load_activity_labels(self):
        """Load activity labels mapping"""
        self.activity_labels = pd.read_csv(
            f'{self.dataset_path}activity_labels.txt',
            sep=' ',
            header=None,
            names=['activity_id', 'activity_name']
        )
        return self.activity_labels

    def load_features(self):
        """Load and clean feature names"""
        self.features = pd.read_csv(
            f'{self.dataset_path}features.txt',
            sep=' ',
            header=None,
            names=['feature_id', 'feature_name']
        )
        # Clean up feature names
        self.features['feature_name'] = (
            self.features['feature_name']
            .str.replace('[()-]', '', regex=True)
            .str.replace(',', '', regex=True)
        )
        return self.features

    @staticmethod
    def get_unique_columns(columns):
        """Make column names unique by appending suffixes"""
        seen = collections.defaultdict(int)
        unique_columns = []
        for col in columns:
            if seen[col] > 0:
                unique_columns.append(f"{col}_{seen[col]}")
            else:
                unique_columns.append(col)
            seen[col] += 1
        return unique_columns

    def load_training_data(self):
        """Load training features and labels"""
        self.X_train = pd.read_csv(
            f'{self.dataset_path}train/X_train.txt',
            sep=r'\s+',
            header=None
        )
        self.y_train = pd.read_csv(
            f'{self.dataset_path}train/y_train.txt',
            sep=r'\s+',
            header=None,
            names=['activity_id']
        )

        # Assign feature names and handle duplicates
        if self.features is not None:
            unique_columns = self.get_unique_columns(self.features['feature_name'])
            self.X_train.columns = unique_columns

        return self.X_train, self.y_train

    def load_test_data(self):
        """Load test features and labels"""
        self.X_test = pd.read_csv(
            f'{self.dataset_path}test/X_test.txt',
            sep=r'\s+',
            header=None
        )
        self.y_test = pd.read_csv(
            f'{self.dataset_path}test/y_test.txt',
            sep=r'\s+',
            header=None,
            names=['activity_id']
        )

        # Assign feature names and handle duplicates
        if self.features is not None:
            unique_columns = self.get_unique_columns(self.features['feature_name'])
            self.X_test.columns = unique_columns

        return self.X_test, self.y_test

    def create_validation_split(self, test_size=0.2, random_state=42):
        """Split training data into train and validation sets"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not loaded. Call load_training_data() first.")

        self.X_train_new, self.X_val, self.y_train_new, self.y_val = train_test_split(
            self.X_train,
            self.y_train,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y_train
        )

        return self.X_train_new, self.X_val, self.y_train_new, self.y_val

    def load_all_data(self):
        """Load all data and prepare for training"""
        if not self.check_dataset_exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.dataset_path}. "
                "Please download it first."
            )

        # Load metadata
        self.load_activity_labels()
        self.load_features()

        # Load data
        self.load_training_data()
        self.load_test_data()

        # Create validation split
        self.create_validation_split()

        return {
            'X_train': self.X_train_new,
            'X_val': self.X_val,
            'X_test': self.X_test,
            'y_train': self.y_train_new,
            'y_val': self.y_val,
            'y_test': self.y_test,
            'activity_labels': self.activity_labels,
            'features': self.features
        }

    def get_data_info(self):
        """Get information about loaded data"""
        info = {}

        if self.X_train_new is not None:
            info['train_samples'] = self.X_train_new.shape[0]
            info['train_features'] = self.X_train_new.shape[1]

        if self.X_val is not None:
            info['val_samples'] = self.X_val.shape[0]

        if self.X_test is not None:
            info['test_samples'] = self.X_test.shape[0]

        if self.activity_labels is not None:
            info['num_classes'] = len(self.activity_labels)
            info['class_names'] = self.activity_labels['activity_name'].tolist()

        return info
