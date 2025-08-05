import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

class WaterQualityPreprocessor:
    """
    Comprehensive data preprocessing class for water quality data.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer_numerical = KNNImputer(n_neighbors=5)
        self.imputer_categorical = SimpleImputer(strategy='most_frequent')
        self.feature_columns = None
        self.target_column = None

    def load_data(self, filepath):
        """Load water quality dataset from CSV file."""
        try:
            data = pd.read_csv(filepath)
            print(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def explore_data(self, data):
        """Basic data exploration and summary statistics."""
        print("Dataset Information:")
        print("=" * 50)
        print(f"Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print("\nData Types:")
        print(data.dtypes)
        print("\nMissing Values:")
        print(data.isnull().sum())
        print("\nStatistical Summary:")
        print(data.describe())
        if 'CCME_WQI' in data.columns:
            print("\nTarget Variable Distribution:")
            print(data['CCME_WQI'].value_counts())

    def handle_missing_values(self, data):
        """
        Handle missing values in the dataset.
        Returns dataset with imputed values.
        """
        data_processed = data.copy()
        numerical_cols = data_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data_processed.select_dtypes(include=['object']).columns.tolist()

        # Remove target column from categorical columns if present
        if 'CCME_WQI' in categorical_cols:
            categorical_cols.remove('CCME_WQI')

        # Impute numerical columns using KNN
        if numerical_cols:
            print(f"Imputing numerical columns: {numerical_cols}")
            data_processed[numerical_cols] = self.imputer_numerical.fit_transform(data_processed[numerical_cols])
        # Impute categorical columns using most frequent
        if categorical_cols:
            print(f"Imputing categorical columns: {categorical_cols}")
            data_processed[categorical_cols] = self.imputer_categorical.fit_transform(data_processed[categorical_cols])

        print(f"Missing values after imputation: {data_processed.isnull().sum().sum()}")
        return data_processed

    def detect_outliers(self, data, columns=None, method='iqr'):
        """
        Detect outliers in numerical columns using IQR or Z-score.
        Return dict: column -> outlier info.
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        outlier_info = {}
        for col in columns:
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            elif method == 'zscore':
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outliers = data[z_scores > 3]
            else:
                outliers = pd.DataFrame()
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': 100 * len(outliers) / len(data),
                'indices': outliers.index.tolist()
            }
        return outlier_info

    def handle_outliers(self, data, columns=None, method='cap'):
        """
        Handle outliers by capping to IQR bounds or removing.
        """
        data_processed = data.copy()
        if columns is None:
            columns = data_processed.select_dtypes(include=[np.number]).columns.tolist()
        for col in columns:
            if method == 'cap':
                Q1 = data_processed[col].quantile(0.25)
                Q3 = data_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data_processed[col] = np.where(data_processed[col] < lower_bound, lower_bound, data_processed[col])
                data_processed[col] = np.where(data_processed[col] > upper_bound, upper_bound, data_processed[col])
        return data_processed

    def preprocess_pipeline(self, data, target_column='CCME_WQI', test_size=0.2, random_state=42):
        """
        Complete preprocessing pipeline.
        Returns processed data (full set, or X_train, X_test, y_train, y_test by adding split).
        """
        print("Starting preprocessing pipeline...")

        # Handle missing values
        print("Step 1: Handling missing values...")
        data = self.handle_missing_values(data)

        # Handle outliers
        print("Step 2: Handling outliers...")
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'CCME_Values' in numerical_cols:  # Remove target if present
            numerical_cols.remove('CCME_Values')
        data = self.handle_outliers(data, columns=numerical_cols, method='cap')

        print("Preprocessing completed successfully!")
        return data

# Example usage
if __name__ == "__main__":
    preprocessor = WaterQualityPreprocessor()
    data = preprocessor.load_data('data/raw/water_quality_comprehensive_dataset.csv')
    if data is not None:
        preprocessor.explore_data(data)
        processed_data = preprocessor.preprocess_pipeline(data)
        print("Preprocessing completed!")
