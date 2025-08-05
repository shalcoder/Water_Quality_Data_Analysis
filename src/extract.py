import pandas as pd

# Load your full dataset
df = pd.read_csv('water_quality_dataset.csv')

# Select only the required columns
required_cols = [
    'Ammonia_mg_L', 'BOD_mg_L', 'DO_mg_L', 'Orthophosphate_mg_L', 
    'pH', 'Temperature_C', 'Nitrogen_mg_L', 'Nitrate_mg_L', 'CCME_Values'
]
batch_data = df[required_cols].head(50)  # Take first 50 rows

# Save for batch prediction
batch_data.to_csv('my_batch_prediction.csv', index=False)
