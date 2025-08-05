"""
setup_project.py
Script to set up the complete project structure and generate the dataset
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_directories():
    """Create all necessary directories"""
    directories = [
        'data/raw',
        'data/processed', 
        'notebooks',
        'src',
        'models',
        'scripts'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def generate_dataset():
    """Generate comprehensive water quality dataset"""
    np.random.seed(42)
    random.seed(42)
    
    # Generate 10,000 realistic water quality samples
    n_samples = 10000
    
    countries = ['USA', 'Canada', 'Ireland', 'England', 'China']
    waterbody_types = ['River', 'Lake', 'Reservoir', 'Stream', 'Creek', 'Pond', 'Canal']
    
    areas = {
        'USA': ['California', 'Texas', 'Florida', 'New York', 'Pennsylvania'],
        'Canada': ['Ontario', 'Quebec', 'British Columbia', 'Alberta'],
        'Ireland': ['Dublin', 'Cork', 'Galway', 'Limerick'],
        'England': ['London', 'Manchester', 'Birmingham', 'Leeds'],
        'China': ['Beijing', 'Shanghai', 'Guangdong', 'Jiangsu']
    }
    
    # Generate dates from 2020 to 2023
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    
    data = []
    
    for i in range(n_samples):
        country = random.choice(countries)
        area = random.choice(areas[country])
        waterbody_type = random.choice(waterbody_types)
        date = random.choice(date_range)
        
        # Generate realistic water quality parameters with correlations
        # pH (6.0 - 9.0)
        ph = np.random.normal(7.2, 0.5)
        ph = np.clip(ph, 6.0, 9.0)
        
        # Temperature (0 - 35°C)
        temperature = np.random.normal(15, 8)
        temperature = np.clip(temperature, 0, 35)
        
        # Dissolved Oxygen (3 - 15 mg/L, inversely related to temperature)
        do_base = 12 - (temperature - 15) * 0.1
        do = np.random.normal(do_base, 1.5)
        do = np.clip(do, 3, 15)
        
        # BOD (0 - 25 mg/L)
        bod = np.random.exponential(2)
        bod = np.clip(bod, 0, 25)
        
        # Ammonia (0 - 2 mg/L)
        ammonia = np.random.exponential(0.1)
        ammonia = np.clip(ammonia, 0, 2)
        
        # Orthophosphate (0 - 0.5 mg/L)
        orthophosphate = np.random.exponential(0.05)
        orthophosphate = np.clip(orthophosphate, 0, 0.5)
        
        # Total Nitrogen (0 - 5 mg/L)
        nitrogen = np.random.exponential(0.8)
        nitrogen = np.clip(nitrogen, 0, 5)
        
        # Nitrate (0 - 10 mg/L)
        nitrate = np.random.exponential(1.5)
        nitrate = np.clip(nitrate, 0, 10)
        
        # Calculate CCME Water Quality Index (0-100)
        ph_score = 100 if 6.5 <= ph <= 8.5 else max(0, 100 - abs(ph - 7.5) * 20)
        do_score = min(100, do * 10) if do >= 6 else max(0, do * 16.67)
        bod_score = max(0, 100 - bod * 5)
        ammonia_score = max(0, 100 - ammonia * 200)
        
        ccme_value = (ph_score + do_score + bod_score + ammonia_score) / 4
        
        # CCME WQI Classification
        if ccme_value >= 95:
            ccme_wqi = 'Excellent'
        elif ccme_value >= 80:
            ccme_wqi = 'Good'
        elif ccme_value >= 65:
            ccme_wqi = 'Fair'
        elif ccme_value >= 45:
            ccme_wqi = 'Marginal'
        else:
            ccme_wqi = 'Poor'
        
        data.append([
            country, area, waterbody_type, date.strftime('%d-%m-%Y'),
            round(ammonia, 3), round(bod, 2), round(do, 2), round(orthophosphate, 3),
            round(ph, 2), round(temperature, 1), round(nitrogen, 2), round(nitrate, 2),
            round(ccme_value, 2), ccme_wqi
        ])
    
    # Create DataFrame
    columns = ['Country', 'Area', 'Waterbody_Type', 'Date', 'Ammonia_mg_L', 'BOD_mg_L', 
               'DO_mg_L', 'Orthophosphate_mg_L', 'pH', 'Temperature_C', 'Nitrogen_mg_L', 
               'Nitrate_mg_L', 'CCME_Values', 'CCME_WQI']
    
    df = pd.DataFrame(data, columns=columns)
    
    # Add some missing values (5% for realism)
    missing_cols = ['Ammonia_mg_L', 'BOD_mg_L', 'Orthophosphate_mg_L', 'Nitrogen_mg_L']
    for col in missing_cols:
        missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[missing_indices, col] = np.nan
    
    # Save dataset
    df.to_csv('data/raw/water_quality_comprehensive_dataset.csv', index=False)
    print(f"✅ Generated dataset with {len(df)} samples")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    
    return df

def main():
    """Main setup function"""
    print("🚀 Setting up Water Quality Prediction Project...")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Generate dataset
    print("\n📊 Generating dataset...")
    df = generate_dataset()
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ Project setup completed successfully!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Train the model: python scripts/train_model.py")
    print("3. Run the app: streamlit run app.py")
    print("\n🎉 Happy coding!")

if __name__ == "__main__":
    main()
