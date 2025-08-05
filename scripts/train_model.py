"""
train_model.py - Windows Compatible, Handles Rare Classes & Label Mismatch

- Fixes class imbalance in your target using SMOTE (n_jobs=1 for Windows)
- Handles rare classes by duplication/augmentation
- Automatically sets correct class labels/target_names for classification_report
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    print("📊 Loading and preprocessing data...")
    df = pd.read_csv('data/raw/water_quality_comprehensive_dataset.csv')
    print(f"Dataset loaded: {df.shape}")

    numeric_cols = ['Ammonia_mg_L', 'BOD_mg_L', 'DO_mg_L', 'Orthophosphate_mg_L',
                    'pH', 'Temperature_C', 'Nitrogen_mg_L', 'Nitrate_mg_L', 'CCME_Values']
    imputer = KNNImputer(n_neighbors=5)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    df['DO_BOD_ratio'] = df['DO_mg_L'] / (df['BOD_mg_L'] + 1e-3)
    df['N_Nitrate_ratio'] = df['Nitrogen_mg_L'] / (df['Nitrate_mg_L'] + 1e-3)

    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    df['Year'] = df['Date'].dt.year.fillna(2023)

    df['pH_Acidic'] = (df['pH'] <= 6.5).astype(int)
    df['pH_Neutral'] = ((df['pH'] > 6.5) & (df['pH'] <= 8.5)).astype(int)
    df['pH_Alkaline'] = (df['pH'] > 8.5).astype(int)

    df['Water_Quality_Score'] = (
        (df['pH_Neutral'] * 25) +
        (np.clip(df['DO_mg_L'] / 15, 0, 1) * 25) +
        (np.clip((25 - df['BOD_mg_L']) / 25, 0, 1) * 25) +
        (np.clip((2 - df['Ammonia_mg_L']) / 2, 0, 1) * 25)
    )

    # Target encoding
    target_map = {'Poor': 0, 'Marginal': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
    df['CCME_WQI_encoded'] = df['CCME_WQI'].map(target_map)

    # ========== RARE CLASS FIX ==========
    class_counts = df['CCME_WQI_encoded'].value_counts().sort_index()
    min_samples_needed = 10  # Set to something >1
    rare_classes = class_counts[class_counts < min_samples_needed].index.tolist()
    if rare_classes:
        print(f"\n⚠️  Fixing rare classes: {rare_classes}")
        for rare_cls in rare_classes:
            rs = df[df['CCME_WQI_encoded'] == rare_cls].copy()
            needed = min_samples_needed - len(rs)
            for _ in range(needed):
                base = rs.sample(1).copy()
                for col in numeric_cols:
                    if col in base.columns:
                        base[col] = base[col] + np.random.normal(0, 0.01)
                df = pd.concat([df, base], ignore_index=True)
        print("✅ Rare class fix: New class counts:")
        print(df['CCME_WQI_encoded'].value_counts().sort_index())
    # ====================================

    feature_cols = [
        'Ammonia_mg_L', 'BOD_mg_L', 'DO_mg_L', 'Orthophosphate_mg_L',
        'pH', 'Temperature_C', 'Nitrogen_mg_L', 'Nitrate_mg_L', 'CCME_Values',
        'DO_BOD_ratio', 'N_Nitrate_ratio', 'Year',
        'pH_Acidic', 'pH_Neutral', 'pH_Alkaline', 'Water_Quality_Score'
    ]

    X = df[feature_cols]
    y = df['CCME_WQI_encoded']

    print(f"\nFinal features: {len(feature_cols)} columns, {len(X):,} samples")
    return X, y, feature_cols

def main():
    print("🚀 Water Quality Model Training (Windows Compatible)")
    print("=" * 60)
    try:
        X, y, feature_names = load_and_preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"\nTrain set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print(f"Scaled feature range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")

        print("\n⚖️ Applying SMOTE class balancing (n_jobs=1 for Windows)...")
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        print(f"Original training set: {X_train_scaled.shape}")
        print(f"Balanced training set: {X_train_balanced.shape}")
        print("Final balanced distribution:")
        for cls, cnt in pd.Series(y_train_balanced).value_counts().sort_index().items():
            print(f"  Class {cls}: {cnt} samples")

        print("\n🤖 Training Random Forest model (n_jobs=1)...")
        model = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2,
            random_state=42, n_jobs=1, class_weight='balanced')
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=cv, scoring='accuracy', n_jobs=1)
        print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"CV Scores: {[f'{score:.3f}' for score in cv_scores]}")
        model.fit(X_train_balanced, y_train_balanced)

        print("\n📈 Evaluating on test set...")
        y_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # === CLASSIFICATION REPORT FIX ===
        unique_classes = sorted(np.unique(y_test))
        class_names_map = {0: 'Poor', 1: 'Marginal', 2: 'Fair', 3: 'Good', 4: 'Excellent'}
        target_names = [class_names_map[cls] for cls in unique_classes]
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, labels=unique_classes, target_names=target_names, digits=4))
        # =================================

        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"\n🎯 Top 10 Features:")
        print(feature_importance.head(10).to_string(index=False))

        # Save artifacts
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/water_quality_model.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(feature_names, 'models/feature_names.pkl')
        feature_importance.to_csv('models/feature_importance.csv', index=False)
        import json
        metadata = {
            'model_type': 'RandomForestClassifier',
            'test_accuracy': float(test_accuracy),
            'n_features': len(feature_names),
            'features': feature_names,
            'training_date': pd.Timestamp.now().isoformat(),
            'windows_compatible': True
        }
        with open('models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print("\n✅ Model, scaler, and metadata saved to models/.")

        print("\n" + "=" * 60)
        print(f"🎉 TRAINING COMPLETED! Final accuracy: {test_accuracy*100:.2f}%")
        print("\nTo run the app: streamlit run app.py")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
