import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
from typing import Dict, List, Union, Optional

warnings.filterwarnings('ignore')

class WaterQualityPredictor:
    """
    Water quality prediction class for making predictions on new data
    """
    
    def __init__(self, model_path: str = "models/water_quality_model.pkl", 
                 scaler_path: str = "models/scaler.pkl"):
        """
        Initialize predictor with model and preprocessing objects
        """
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.n_classes = None
        
        # Load model and scaler
        if Path(model_path).exists():
            self.load_model(model_path)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        if Path(scaler_path).exists():
            self.load_scaler(scaler_path)
        else:
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    def load_model(self, model_path: str):
        """Load trained model from file"""
        try:
            self.model = joblib.load(model_path)
            
            # Determine number of classes from the model
            if hasattr(self.model, 'n_classes_'):
                self.n_classes = self.model.n_classes_
            elif hasattr(self.model, 'classes_'):
                self.n_classes = len(self.model.classes_)
            else:
                self.n_classes = 5  # Default assumption
            
            print(f"Model loaded successfully from {model_path}")
            print(f"Model expects {self.n_classes} classes")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def load_scaler(self, scaler_path: str):
        """Load fitted scaler from file"""
        try:
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded successfully from {scaler_path}")
            
            # Get expected feature names from scaler
            if hasattr(self.scaler, 'feature_names_in_'):
                self.feature_names = list(self.scaler.feature_names_in_)
                print(f"Expected features ({len(self.feature_names)}): {self.feature_names}")
            else:
                print(f"Scaler expects {self.scaler.n_features_in_} features")
                
        except Exception as e:
            print(f"Error loading scaler: {e}")
            raise

    def _prepare_features(self, input_data: Dict[str, float]) -> np.ndarray:
        """
        Prepare features in the exact order and format expected by the scaler
        """
        # Base features with defaults
        base_features = {
            'Ammonia_mg_L': float(input_data.get('Ammonia_mg_L', 0.0)),
            'BOD_mg_L': float(input_data.get('BOD_mg_L', 2.0)),
            'DO_mg_L': float(input_data.get('DO_mg_L', 8.0)),
            'Orthophosphate_mg_L': float(input_data.get('Orthophosphate_mg_L', 0.05)),
            'pH': float(input_data.get('pH', 7.0)),
            'Temperature_C': float(input_data.get('Temperature_C', 15.0)),
            'Nitrogen_mg_L': float(input_data.get('Nitrogen_mg_L', 1.0)),
            'Nitrate_mg_L': float(input_data.get('Nitrate_mg_L', 1.0)),
            'CCME_Values': float(input_data.get('CCME_Values', 85.0)),
        }
        
        # Engineered features
        do_bod_ratio = base_features['DO_mg_L'] / (base_features['BOD_mg_L'] + 1e-3)
        n_nitrate_ratio = base_features['Nitrogen_mg_L'] / (base_features['Nitrate_mg_L'] + 1e-3)
        
        ph_acidic = 1 if base_features['pH'] <= 6.5 else 0
        ph_neutral = 1 if 6.5 < base_features['pH'] <= 8.5 else 0
        ph_alkaline = 1 if base_features['pH'] > 8.5 else 0
        
        # Calculate Water_Quality_Score (likely missing feature)
        water_quality_score = (
            (ph_neutral * 25) +
            (np.clip(base_features['DO_mg_L'] / 15, 0, 1) * 25) +
            (np.clip((25 - base_features['BOD_mg_L']) / 25, 0, 1) * 25) +
            (np.clip((2 - base_features['Ammonia_mg_L']) / 2, 0, 1) * 25)
        )
        
        engineered_features = {
            'DO_BOD_ratio': do_bod_ratio,
            'N_Nitrate_ratio': n_nitrate_ratio,
            'Year': 2023,
            'pH_Acidic': ph_acidic,
            'pH_Neutral': ph_neutral,
            'pH_Alkaline': ph_alkaline,
            'Water_Quality_Score': water_quality_score
        }
        
        # Combine all features
        all_features = {**base_features, **engineered_features}
        
        # If we have feature names from scaler, use them to order features correctly
        if self.feature_names:
            feature_values = []
            for feature_name in self.feature_names:
                if feature_name in all_features:
                    feature_values.append(all_features[feature_name])
                else:
                    # Add default value for missing features
                    feature_values.append(0.0)
                    print(f"Warning: Feature '{feature_name}' not found, using default value 0.0")
            
            return np.array(feature_values).reshape(1, -1)
        else:
            # Standard 16-feature order (most likely based on your training)
            feature_values = [
                all_features['Ammonia_mg_L'],
                all_features['BOD_mg_L'],
                all_features['DO_mg_L'],
                all_features['Orthophosphate_mg_L'],
                all_features['pH'],
                all_features['Temperature_C'],
                all_features['Nitrogen_mg_L'],
                all_features['Nitrate_mg_L'],
                all_features['CCME_Values'],
                all_features['DO_BOD_ratio'],
                all_features['N_Nitrate_ratio'],
                all_features['Year'],
                all_features['pH_Acidic'],
                all_features['pH_Neutral'],
                all_features['pH_Alkaline'],
                all_features['Water_Quality_Score']
            ]
            
            return np.array(feature_values).reshape(1, -1)

    def predict_single(self, input_data: Dict[str, float]) -> Dict:
        """
        Make prediction for a single water sample
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        # Prepare features
        features_array = self._prepare_features(input_data)
        
        print(f"Features shape: {features_array.shape}")
        print(f"Expected features: {self.scaler.n_features_in_}")
        
        # Scale features
        if self.scaler is not None:
            features_array = self.scaler.transform(features_array)
        
        # Make prediction
        prediction = self.model.predict(features_array)[0]
        print(f"Raw prediction: {prediction}")
        print(f"Prediction type: {type(prediction)}")
        print(f"Model classes: {getattr(self.model, 'classes_', 'No classes attribute')}")
        
        # Get probability if available
        prediction_proba = None
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_array)[0]
            prediction_proba = probabilities.tolist()
            print(f"Probabilities shape: {probabilities.shape}")
            print(f"Probabilities: {probabilities}")
        
        # Convert prediction back to original labels with bounds checking
        # Dynamic quality mapping based on actual number of classes
        if self.n_classes == 4:
            quality_map = {0: 'Poor', 1: 'Fair', 2: 'Good', 3: 'Excellent'}
        elif self.n_classes == 5:
            quality_map = {0: 'Poor', 1: 'Marginal', 2: 'Fair', 3: 'Good', 4: 'Excellent'}
        else:
            # Fallback for other cases
            quality_labels = ['Poor', 'Marginal', 'Fair', 'Good', 'Excellent']
            quality_map = {i: quality_labels[min(i, len(quality_labels)-1)] for i in range(self.n_classes)}
        
        # Ensure prediction is within bounds
        prediction = max(0, min(prediction, len(quality_map) - 1))
        prediction_label = quality_map.get(prediction, 'Unknown')
        
        return {
            'prediction': prediction_label,
            'prediction_numeric': prediction,
            'probabilities': prediction_proba,
            'input_parameters': input_data
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for multiple water samples
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        # Prepare all features at once for efficiency
        processed_rows = []
        
        for idx, row in df.iterrows():
            input_data = row.to_dict()
            try:
                # Convert any NaN values to defaults
                for key, value in input_data.items():
                    if pd.isna(value):
                        input_data[key] = 0.0
                
                features_array = self._prepare_features(input_data)
                processed_rows.append(features_array[0])  # Remove the extra dimension
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                # Add a default feature vector
                default_features = np.zeros(self.scaler.n_features_in_)
                processed_rows.append(default_features)
        
        # Convert to numpy array and scale
        all_features = np.array(processed_rows)
        print(f"Batch features shape: {all_features.shape}")
        
        if self.scaler is not None:
            all_features = self.scaler.transform(all_features)
        
        # Make predictions
        predictions = self.model.predict(all_features)
        print(f"Predictions range: {np.min(predictions)} to {np.max(predictions)}")
        print(f"Unique predictions: {np.unique(predictions)}")
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(all_features)
            print(f"Probabilities shape: {probabilities.shape}")
            # Safe confidence calculation with bounds checking
            confidences = []
            for i, pred in enumerate(predictions):
                if 0 <= pred < probabilities.shape[1]:
                    confidences.append(probabilities[i][pred])
                else:
                    print(f"Warning: Prediction {pred} out of bounds for probabilities")
                    confidences.append(0.0)
        else:
            confidences = [1.0] * len(predictions)
        
        # Add results to original dataframe
        result_df = df.copy()
        
        # Dynamic quality mapping based on actual number of classes
        if self.n_classes == 4:
            quality_map = {0: 'Poor', 1: 'Fair', 2: 'Good', 3: 'Excellent'}
        elif self.n_classes == 5:
            quality_map = {0: 'Poor', 1: 'Marginal', 2: 'Fair', 3: 'Good', 4: 'Excellent'}
        else:
            # Fallback for other cases
            quality_labels = ['Poor', 'Marginal', 'Fair', 'Good', 'Excellent']
            quality_map = {i: quality_labels[min(i, len(quality_labels)-1)] for i in range(self.n_classes)}
        
        # Safe prediction mapping with bounds checking
        predicted_qualities = []
        for pred in predictions:
            pred = max(0, min(pred, len(quality_map) - 1))  # Ensure within bounds
            predicted_qualities.append(quality_map.get(pred, 'Unknown'))
        
        result_df['Predicted_Quality'] = predicted_qualities
        result_df['Confidence'] = confidences
        
        return result_df

    def predict_water_quality_index(self, input_data: Dict[str, float]) -> Dict:
        """
        Predict water quality index and provide interpretation
        """
        prediction_result = self.predict_single(input_data)
        
        # Interpretation based on CCME Water Quality Index
        quality_interpretations = {
            'Excellent': {
                'description': 'Water quality is protected with a virtual absence of threat or impairment.',
                'color_code': 'green',
                'safety': 'Safe for all uses'
            },
            'Good': {
                'description': 'Water quality is protected with only a minor degree of threat or impairment.',
                'color_code': 'blue',
                'safety': 'Safe for most uses'
            },
            'Fair': {
                'description': 'Water quality is usually protected but occasionally threatened or impaired.',
                'color_code': 'yellow',
                'safety': 'May require treatment for some uses'
            },
            'Marginal': {
                'description': 'Water quality is frequently threatened or impaired.',
                'color_code': 'orange',
                'safety': 'Requires treatment for most uses'
            },
            'Poor': {
                'description': 'Water quality is almost always threatened or impaired.',
                'color_code': 'red',
                'safety': 'Not safe for use without significant treatment'
            }
        }
        
        predicted_class = prediction_result['prediction']
        interpretation = quality_interpretations.get(predicted_class, {})
        
        return {
            'predicted_quality': predicted_class,
            'quality_description': interpretation.get('description', 'Unknown'),
            'safety_level': interpretation.get('safety', 'Unknown'),
            'color_code': interpretation.get('color_code', 'gray'),
            'confidence': prediction_result.get('probabilities'),
            'input_parameters': input_data
        }

# Example usage
if __name__ == "__main__":
    predictor = WaterQualityPredictor()
    
    sample_input = {
        'Ammonia_mg_L': 0.05,
        'BOD_mg_L': 2.1,
        'DO_mg_L': 8.5,
        'Orthophosphate_mg_L': 0.03,
        'pH': 7.2,
        'Temperature_C': 18.5,
        'Nitrogen_mg_L': 0.8,
        'Nitrate_mg_L': 1.2,
        'CCME_Values': 85
    }
    
    try:
        result = predictor.predict_water_quality_index(sample_input)
        print("Prediction Result:", result['predicted_quality'])
    except Exception as e:
        print(f"Error: {e}")
