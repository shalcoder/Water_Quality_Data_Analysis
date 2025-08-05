"""
evaluate_model.py
Script to evaluate the saved model on the test set
"""

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    """Main evaluation function"""
    print("📈 Evaluating Water Quality Model...")
    print("=" * 50)
    
    try:
        # Load test data (you need to create this during preprocessing)
        # For now, we'll use a portion of the original dataset
        df = pd.read_csv('data/raw/water_quality_comprehensive_dataset.csv')
        
        # Basic preprocessing (same as training)
        from sklearn.impute import KNNImputer
        from sklearn.preprocessing import RobustScaler
        from sklearn.model_selection import train_test_split
        
        numeric_cols = ['Ammonia_mg_L', 'BOD_mg_L', 'DO_mg_L', 'Orthophosphate_mg_L',
                       'pH', 'Temperature_C', 'Nitrogen_mg_L', 'Nitrate_mg_L', 'CCME_Values']
        
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        # Feature engineering
        df['DO_BOD_ratio'] = df['DO_mg_L'] / (df['BOD_mg_L'] + 1e-3)
        df['N_Nitrate_ratio'] = df['Nitrogen_mg_L'] / (df['Nitrate_mg_L'] + 1e-3)
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        df['Year'] = df['Date'].dt.year
        df['pH_Acidic'] = (df['pH'] <= 6.5).astype(int)
        df['pH_Neutral'] = ((df['pH'] > 6.5) & (df['pH'] <= 8.5)).astype(int)
        df['pH_Alkaline'] = (df['pH'] > 8.5).astype(int)
        
        # Encode target
        target_map = {'Poor': 0, 'Marginal': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
        df['CCME_WQI_encoded'] = df['CCME_WQI'].map(target_map)
        
        # Select features
        feature_cols = ['Ammonia_mg_L', 'BOD_mg_L', 'DO_mg_L', 'Orthophosphate_mg_L',
                       'pH', 'Temperature_C', 'Nitrogen_mg_L', 'Nitrate_mg_L', 'CCME_Values',
                       'DO_BOD_ratio', 'N_Nitrate_ratio', 'Year', 'pH_Acidic', 'pH_Neutral', 'pH_Alkaline']
        
        X = df[feature_cols]
        y = df['CCME_WQI_encoded']
        
        # Use same split as training
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Load model and scaler
        model = joblib.load('models/water_quality_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        # Scale test data
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\n=== Classification Report ===")
        target_names = ['Poor', 'Marginal', 'Fair', 'Good', 'Excellent']
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix - Water Quality Prediction')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\n=== Top 10 Feature Importances ===")
            print(importance_df.head(10))
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df.head(10), y='Feature', x='Importance')
            plt.title('Top 10 Feature Importances')
            plt.tight_layout()
            plt.show()
        
        print("\n✅ Model evaluation completed!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run 'python scripts/train_model.py' first!")
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()
