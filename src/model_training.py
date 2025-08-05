import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
warnings.filterwarnings('ignore')

class WaterQualityModelTrainer:
    """
    Comprehensive model training class for water quality prediction
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.model_performance = {}
        
    def initialize_models(self):
        """
        Initialize different machine learning models
        """
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=self.random_state,
                probability=True
            ),
            'LogisticRegression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.random_state
            )
        }
    
    def train_individual_models(self, X_train, y_train, cv_folds=5):
        """
        Train individual models and evaluate using cross-validation
        
        Parameters:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        cv_folds (int): Number of cross-validation folds
        
        Returns:
        dict: Model performance results
        """
        self.initialize_models()
        
        print("Training individual models...")
        print("=" * 50)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            
            # Fit the model
            model.fit(X_train, y_train)
            
            # Store performance
            self.model_performance[name] = {
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model
            }
            
            print(f"{name} - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.model_performance
    
    def save_model(self, model, filepath):
        """
        Save trained model to file
        
        Parameters:
        model: Trained model
        filepath (str): Path to save the model
        """
        try:
            joblib.dump(model, filepath)
            print(f"Model saved successfully to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filepath):
        """
        Load trained model from file
        
        Parameters:
        filepath (str): Path to the saved model
        
        Returns:
        sklearn model: Loaded model
        """
        try:
            model = joblib.load(filepath)
            print(f"Model loaded successfully from {filepath}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

# Example usage
if __name__ == "__main__":
    trainer = WaterQualityModelTrainer(random_state=42)
    print("Model trainer initialized successfully!")
