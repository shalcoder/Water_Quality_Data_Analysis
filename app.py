import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from prediction import WaterQualityPredictor
except ImportError:
    st.error("Error importing prediction module. Please check your src directory.")

# Page configuration
st.set_page_config(
    page_title="🌊 Water Quality AI Analytics",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Dark Theme CSS
st.markdown("""
<style>
    /* Global Dark Theme */
    .reportview-container {
        background: linear-gradient(135deg, #0f1419 0%, #1a202c 100%);
        color: #e2e8f0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
        color: #e2e8f0;
    }
    
    /* Custom Cards */
    .metric-card {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #4a5568;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.4);
    }
    
    .success-metric {
        background: linear-gradient(135deg, #22543d 0%, #2f855a 100%);
        border: 1px solid #38a169;
        color: #f0fff4;
    }
    
    .warning-metric {
        background: linear-gradient(135deg, #744210 0%, #d69e2e 100%);
        border: 1px solid #ecc94b;
        color: #fffbeb;
    }
    
    .danger-metric {
        background: linear-gradient(135deg, #742a2a 0%, #e53e3e 100%);
        border: 1px solid #fc8181;
        color: #fff5f5;
    }
    
    .info-metric {
        background: linear-gradient(135deg, #2c5282 0%, #3182ce 100%);
        border: 1px solid #63b3ed;
        color: #ebf8ff;
    }
    
    /* Enhanced Typography */
    .big-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(45deg, #4299e1, #38b2ac);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #63b3ed;
        margin: 2rem 0 1rem 0;
        border-bottom: 2px solid #4a5568;
        padding-bottom: 0.5rem;
    }
    
    /* Input Styling */
    .stNumberInput > div > div > input {
        background-color: #2d3748;
        color: #e2e8f0;
        border: 1px solid #4a5568;
        border-radius: 8px;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #4299e1, #38b2ac);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(66, 153, 225, 0.4);
    }
    
    /* Plotly Dark Theme */
    .js-plotly-plot .plotly .modebar {
        background-color: #2d3748 !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(45deg, #4299e1, #38b2ac);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content .stRadio > div {
        background-color: #2d3748;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Custom animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load sample data for exploration"""
    try:
        df = pd.read_csv('data/raw/water_quality_comprehensive_dataset.csv')
        # Parse dates if present
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
            except:
                pass
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please check the file path.")
        return None

@st.cache_resource
def load_predictor():
    """Load the predictor (cached)"""
    try:
        return WaterQualityPredictor()
    except Exception as e:
        st.error(f"Error loading predictor: {e}")
        return None

def create_advanced_plots(data):
    """Create advanced visualization plots"""
    
    # 1. Correlation Matrix with Clustering
    st.subheader("🔗 Advanced Correlation Analysis")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr = data[numeric_cols].corr()
        
        # Create clustered heatmap
        fig = px.imshow(
            corr, 
            text_auto=True, 
            aspect="auto",
            title="Feature Correlation Matrix with Clustering",
            color_continuous_scale='RdBu_r',
            template='plotly_dark'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 2. Pairplot for key features
    st.subheader("📊 Multi-Dimensional Analysis")
    key_features = ['pH', 'DO_mg_L', 'BOD_mg_L', 'Ammonia_mg_L', 'CCME_Values']
    available_features = [f for f in key_features if f in data.columns]
    
    if len(available_features) >= 3 and 'CCME_WQI' in data.columns:
        # Sample data for performance
        sample_data = data[available_features + ['CCME_WQI']].sample(min(500, len(data)))
        
        # Create scatter matrix
        try:
            fig = ff.create_scatterplotmatrix(
                sample_data[available_features], 
                diag='histogram',
                height=600, 
                width=800,
                title="Parameter Relationships Matrix"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                template='plotly_dark'
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create scatter matrix: {str(e)}")
    
    # 3. Box plots for outlier detection
    st.subheader("📦 Outlier Detection Analysis")
    
    cols = st.columns(2)
    for i, feature in enumerate(available_features[:4]):
        with cols[i % 2]:
            try:
                fig = px.box(
                    data, 
                    y=feature,
                    title=f'{feature} Distribution & Outliers',
                    template='plotly_dark'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create box plot for {feature}: {str(e)}")
    
    # 4. Time series analysis if date available
    if 'Date' in data.columns and data['Date'].notna().sum() > 10:
        st.subheader("📈 Temporal Analysis")
        
        # Group by month and calculate average CCME_Values
        if 'CCME_Values' in data.columns:
            try:
                monthly_data = data.groupby(data['Date'].dt.to_period('M'))['CCME_Values'].mean().reset_index()
                monthly_data['Date'] = monthly_data['Date'].dt.to_timestamp()
                
                fig = px.line(
                    monthly_data, 
                    x='Date', 
                    y='CCME_Values',
                    title='Water Quality Trends Over Time',
                    template='plotly_dark'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create time series plot: {str(e)}")
    
    # 5. Fixed Parallel coordinates plot
    if 'CCME_WQI' in data.columns and len(available_features) >= 3:
        st.subheader("🎯 Multi-Parameter Quality Analysis")
        
        try:
            # Sample and prepare data
            sample_data = data[available_features + ['CCME_WQI']].dropna().sample(min(200, len(data)))
            
            # Ensure we have enough data
            if len(sample_data) < 10:
                st.warning("Not enough data points for meaningful parallel coordinates visualization.")
            else:
                # Convert categorical quality labels to numeric values for coloring
                quality_map = {'Poor': 0, 'Marginal': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
                sample_data = sample_data.copy()  # Avoid SettingWithCopyWarning
                sample_data['quality_numeric'] = sample_data['CCME_WQI'].map(quality_map)
                
                # Check for unmapped values
                unmapped_count = sample_data['quality_numeric'].isna().sum()
                if unmapped_count > 0:
                    st.info(f"Note: {unmapped_count} samples with unrecognized quality labels were excluded.")
                
                # Remove any unmapped values
                sample_data = sample_data.dropna(subset=['quality_numeric'])
                
                if len(sample_data) > 0:
                    fig = px.parallel_coordinates(
                        sample_data,
                        color='quality_numeric',  # Use numeric values instead of categorical
                        dimensions=available_features,
                        title="Parameter Interactions by Water Quality",
                        template='plotly_dark',
                        color_continuous_scale='RdYlGn',  # Red-Yellow-Green scale
                        labels={'quality_numeric': 'Quality Score'}
                    )
                    
                    # Customize the color bar to show quality labels
                    fig.update_coloraxes(
                        colorbar=dict(
                            title="Water Quality Level",
                            tickvals=[0, 1, 2, 3, 4],
                            ticktext=['Poor', 'Marginal', 'Fair', 'Good', 'Excellent'],
                            len=0.8
                        )
                    )
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add interpretation
                    st.info("""
                    **How to read this chart:**
                    - Each vertical line represents a water quality parameter
                    - Each colored line represents one water sample
                    - Colors indicate quality: Red (Poor) → Yellow (Fair) → Green (Excellent)
                    - Parallel patterns suggest parameter correlations
                    """)
                else:
                    st.warning("No valid data available after quality mapping.")
                    
        except Exception as e:
            st.error(f"Error creating parallel coordinates plot: {str(e)}")
            st.info("This visualization requires numeric data for all selected parameters.")

def create_geographic_analysis(data):
    """Create geographic analysis if location data available"""
    if 'Country' in data.columns:
        st.subheader("🌍 Geographic Quality Distribution")
        
        try:
            # Country-wise analysis
            country_stats = data.groupby('Country').agg({
                'CCME_Values': ['mean', 'count'],
                'pH': 'mean',
                'DO_mg_L': 'mean'
            }).round(2)
            
            country_stats.columns = ['Avg_CCME', 'Sample_Count', 'Avg_pH', 'Avg_DO']
            country_stats = country_stats.reset_index()
            
            # Create choropleth-style bar chart
            fig = px.bar(
                country_stats, 
                x='Country', 
                y='Avg_CCME',
                color='Avg_CCME',
                size='Sample_Count',
                title='Average Water Quality by Country',
                template='plotly_dark',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display country statistics
            st.dataframe(country_stats.style.background_gradient(subset=['Avg_CCME']))
        except Exception as e:
            st.warning(f"Could not create geographic analysis: {str(e)}")

def show_model_performance():
    """Show model performance metrics"""
    st.subheader("🎯 Model Performance Analysis")
    
    # Mock performance data (replace with actual model evaluation)
    performance_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [0.92, 0.91, 0.90, 0.91],
        'Class_Poor': [0.85, 0.88, 0.82, 0.85],
        'Class_Fair': [0.89, 0.87, 0.91, 0.89],
        'Class_Good': [0.94, 0.93, 0.95, 0.94],
        'Class_Excellent': [0.96, 0.95, 0.97, 0.96]
    }
    
    perf_df = pd.DataFrame(performance_data)
    
    # Create performance visualization
    fig = go.Figure()
    
    for col in ['Class_Poor', 'Class_Fair', 'Class_Good', 'Class_Excellent']:
        fig.add_trace(go.Scatter(
            x=perf_df['Metric'],
            y=perf_df[col],
            mode='lines+markers',
            name=col.replace('Class_', ''),
            line=dict(width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title='Model Performance by Quality Class',
        xaxis_title='Metrics',
        yaxis_title='Score',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (mock data)
    st.subheader("🔍 Feature Importance Analysis")
    
    importance_data = {
        'Feature': ['CCME_Values', 'DO_mg_L', 'pH', 'BOD_mg_L', 'Ammonia_mg_L', 
                   'Temperature_C', 'Nitrogen_mg_L', 'Nitrate_mg_L', 'Orthophosphate_mg_L'],
        'Importance': [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02]
    }
    
    imp_df = pd.DataFrame(importance_data)
    
    fig = px.bar(
        imp_df, 
        x='Importance', 
        y='Feature',
        orientation='h',
        title='Feature Importance Ranking',
        template='plotly_dark',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig, use_container_width=True)

def show_single_prediction():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🔍 Single Sample Prediction</h2>', unsafe_allow_html=True)
    
    # Load predictor
    predictor = load_predictor()
    if predictor is None:
        st.error("Could not load the prediction model. Please check model files.")
        return
    
    st.write("### Enter Water Quality Parameters:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.01, 
                            help="pH level of water (0-14 scale)")
        ammonia = st.number_input("Ammonia (mg/L)", min_value=0.0, max_value=10.0, value=0.1, step=0.01,
                                 help="Ammonia concentration in mg/L")
        bod = st.number_input("BOD (mg/L)", min_value=0.0, max_value=50.0, value=2.0, step=0.01,
                             help="Biochemical Oxygen Demand in mg/L")
    
    with col2:
        do = st.number_input("Dissolved Oxygen (mg/L)", min_value=0.0, max_value=20.0, value=8.0, step=0.01,
                            help="Dissolved oxygen concentration in mg/L")
        orthophosphate = st.number_input("Orthophosphate (mg/L)", min_value=0.0, max_value=10.0, value=0.05, step=0.01,
                                        help="Orthophosphate concentration in mg/L")
        temperature = st.number_input("Temperature (°C)", min_value=-5.0, max_value=40.0, value=15.0, step=0.1,
                                     help="Water temperature in Celsius")
    
    with col3:
        nitrogen = st.number_input("Nitrogen (mg/L)", min_value=0.0, max_value=20.0, value=1.0, step=0.01,
                                  help="Total nitrogen concentration in mg/L")
        nitrate = st.number_input("Nitrate (mg/L)", min_value=0.0, max_value=20.0, value=1.0, step=0.01,
                                 help="Nitrate concentration in mg/L")
        ccme_values = st.number_input("CCME Values", min_value=0.0, max_value=100.0, value=85.0, step=0.1,
                                     help="Canadian Council of Ministers of the Environment values")
    
    if st.button("🔮 Predict Water Quality", type="primary"):
        input_params = {
            'pH': ph,
            'Ammonia_mg_L': ammonia,
            'BOD_mg_L': bod,
            'DO_mg_L': do,
            'Orthophosphate_mg_L': orthophosphate,
            'Temperature_C': temperature,
            'Nitrogen_mg_L': nitrogen,
            'Nitrate_mg_L': nitrate,
            'CCME_Values': ccme_values
        }
        
        try:
            with st.spinner("🧠 AI is analyzing water quality..."):
                result = predictor.predict_water_quality_index(input_params)
                
            quality = result['predicted_quality']
            probabilities = result.get('confidence', [])
            
            # Display results with enhanced styling
            st.markdown("---")
            st.markdown('<h3 class="section-header">🎯 Prediction Results</h3>', unsafe_allow_html=True)
            
            # Main result with appropriate styling
            if quality in ['Excellent', 'Good']:
                st.markdown(f"""
                <div class="metric-card success-metric">
                    <h2>✅ {quality}</h2>
                    <p><strong>Safety Level:</strong> {result.get('safety_level', 'Unknown')}</p>
                    <p><strong>Description:</strong> {result.get('quality_description', 'No description available')}</p>
                </div>
                """, unsafe_allow_html=True)
            elif quality in ['Fair']:
                st.markdown(f"""
                <div class="metric-card warning-metric">
                    <h2>⚠️ {quality}</h2>
                    <p><strong>Safety Level:</strong> {result.get('safety_level', 'Unknown')}</p>
                    <p><strong>Description:</strong> {result.get('quality_description', 'No description available')}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card danger-metric">
                    <h2>❌ {quality}</h2>
                    <p><strong>Safety Level:</strong> {result.get('safety_level', 'Unknown')}</p>
                    <p><strong>Description:</strong> {result.get('quality_description', 'No description available')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Enhanced probability visualization
            if probabilities and len(probabilities) >= 4:
                st.markdown('<h3 class="section-header">📊 Prediction Confidence</h3>', unsafe_allow_html=True)
                
                # Determine quality labels based on number of classes
                if len(probabilities) == 4:
                    quality_labels = ['Poor', 'Fair', 'Good', 'Excellent']
                else:
                    quality_labels = ['Poor', 'Marginal', 'Fair', 'Good', 'Excellent'][:len(probabilities)]
                
                prob_df = pd.DataFrame({
                    'Quality': quality_labels,
                    'Probability': probabilities[:len(quality_labels)]
                })
                
                # Create an enhanced bar chart
                fig = px.bar(
                    prob_df, 
                    x='Quality', 
                    y='Probability',
                    title='AI Confidence Distribution',
                    color='Probability',
                    color_continuous_scale='Viridis',
                    template='plotly_dark'
                )
                
                fig.update_traces(
                    texttemplate='%{y:.1%}', 
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1%}<extra></extra>'
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence metrics
                col1, col2, col3, col4 = st.columns(4)
                for i, (label, prob) in enumerate(zip(quality_labels, probabilities[:len(quality_labels)])):
                    with [col1, col2, col3, col4][i % 4]:
                        st.metric(label, f"{prob:.1%}")
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            with st.expander("Show error details"):
                st.code(str(e))
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_batch_prediction():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📊 Batch Prediction Analytics</h2>', unsafe_allow_html=True)
    
    # Load predictor
    predictor = load_predictor()
    if predictor is None:
        st.error("Could not load the prediction model. Please check model files.")
        return
    
    st.markdown("""
    ### 📁 Upload Your Dataset
    
    Upload a CSV file with water quality parameters for comprehensive batch analysis.
    
    **Required columns:**
    `Ammonia_mg_L`, `BOD_mg_L`, `DO_mg_L`, `Orthophosphate_mg_L`, `pH`, `Temperature_C`, `Nitrogen_mg_L`, `Nitrate_mg_L`, `CCME_Values`
    """)
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Enhanced data preview
            st.markdown('<h3 class="section-header">📋 Dataset Overview</h3>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Samples", f"{len(df):,}")
            with col2:
                st.metric("📈 Features", len(df.columns))
            with col3:
                st.metric("💾 File Size", f"{uploaded_file.size / 1024:.1f} KB")
            with col4:
                st.metric("❓ Missing Values", f"{df.isnull().sum().sum():,}")
            
            # Data preview with styling
            st.dataframe(
                df.head(10).style.background_gradient(subset=df.select_dtypes(include=[np.number]).columns.tolist()),
                use_container_width=True
            )
            
            required_cols = [
                'Ammonia_mg_L', 'BOD_mg_L', 'DO_mg_L', 'Orthophosphate_mg_L', 
                'pH', 'Temperature_C', 'Nitrogen_mg_L', 'Nitrate_mg_L', 'CCME_Values'
            ]
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                with st.expander("📋 Show all required columns"):
                    st.write(required_cols)
            else:
                st.success("✅ All required columns are present!")
                
                if st.button("🚀 Run Advanced Batch Analysis", type="primary"):
                    try:
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text('🔄 Preparing features...')
                        progress_bar.progress(20)
                        
                        status_text.text('🧠 Running AI predictions...')
                        progress_bar.progress(50)
                        
                        result_df = predictor.predict_batch(df)
                        
                        progress_bar.progress(80)
                        status_text.text('📊 Generating analytics...')
                        
                        progress_bar.progress(100)
                        status_text.text('✅ Analysis completed!')
                        
                        # Enhanced Results Display
                        st.markdown('<h3 class="section-header">📈 Prediction Results & Analytics</h3>', unsafe_allow_html=True)
                        
                        # Results summary cards
                        if 'Predicted_Quality' in result_df.columns:
                            quality_counts = result_df['Predicted_Quality'].value_counts()
                            
                            # Summary metrics
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            colors = {
                                'Excellent': 'success-metric',
                                'Good': 'info-metric', 
                                'Fair': 'warning-metric',
                                'Marginal': 'warning-metric',
                                'Poor': 'danger-metric'
                            }
                            
                            for i, (quality, count) in enumerate(quality_counts.items()):
                                percentage = (count / len(result_df)) * 100
                                card_class = colors.get(quality, 'metric-card')
                                
                                with [col1, col2, col3, col4, col5][i % 5]:
                                    st.markdown(f"""
                                    <div class="metric-card {card_class}">
                                        <h3>{quality}</h3>
                                        <h2>{count}</h2>
                                        <p>{percentage:.1f}%</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # Enhanced visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Quality distribution pie chart
                            fig = px.pie(
                                values=quality_counts.values, 
                                names=quality_counts.index,
                                title="🎯 Water Quality Distribution",
                                template='plotly_dark',
                                color_discrete_sequence=px.colors.qualitative.Set3
                            )
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color='white'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Confidence distribution
                            if 'Confidence' in result_df.columns:
                                fig = px.histogram(
                                    result_df, 
                                    x='Confidence',
                                    title="📊 Prediction Confidence Distribution",
                                    template='plotly_dark',
                                    nbins=20
                                )
                                fig.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font_color='white'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Parameter analysis
                        st.markdown('<h3 class="section-header">🔬 Parameter Analysis by Quality</h3>', unsafe_allow_html=True)
                        
                        # Box plots for key parameters by quality
                        key_params = ['pH', 'DO_mg_L', 'BOD_mg_L', 'Ammonia_mg_L']
                        available_params = [p for p in key_params if p in result_df.columns]
                        
                        if available_params:
                            fig = make_subplots(
                                rows=2, cols=2,
                                subplot_titles=available_params[:4],
                                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                       [{"secondary_y": False}, {"secondary_y": False}]]
                            )
                            
                            for i, param in enumerate(available_params[:4]):
                                row = i // 2 + 1
                                col = i % 2 + 1
                                
                                for quality in result_df['Predicted_Quality'].unique():
                                    data_subset = result_df[result_df['Predicted_Quality'] == quality][param]
                                    fig.add_box(
                                        y=data_subset,
                                        name=f"{quality}",
                                        row=row, col=col,
                                        showlegend=(i == 0)
                                    )
                            
                            fig.update_layout(
                                height=600,
                                title_text="Parameter Distributions by Water Quality",
                                template='plotly_dark',
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color='white'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Results table with enhanced styling
                        st.markdown('<h3 class="section-header">📋 Detailed Results</h3>', unsafe_allow_html=True)
                        
                        # Color coding function
                        def highlight_quality(val):
                            if val == 'Excellent':
                                return 'background-color: #22543d; color: white'
                            elif val == 'Good':
                                return 'background-color: #2c5282; color: white'
                            elif val == 'Fair':
                                return 'background-color: #744210; color: white'
                            elif val == 'Marginal':
                                return 'background-color: #d69e2e; color: black'
                            elif val == 'Poor':
                                return 'background-color: #742a2a; color: white'
                            return ''
                        
                        styled_df = result_df.style.applymap(
                            highlight_quality, 
                            subset=['Predicted_Quality']
                        ).format({'Confidence': '{:.2%}'})
                        
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Download section
                        st.markdown('<h3 class="section-header">💾 Download Results</h3>', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Full Results (CSV)",
                                data=csv,
                                file_name=f"water_quality_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # Summary report
                            summary_report = f"""
# Water Quality Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Summary
- Total Samples: {len(result_df):,}
- Features Analyzed: {len(required_cols)}

## Quality Distribution
{quality_counts.to_string()}

## Key Insights
- Most Common Quality: {quality_counts.index[0]}
- Average Confidence: {result_df['Confidence'].mean():.2%}
- Samples Requiring Attention: {quality_counts.get('Poor', 0) + quality_counts.get('Marginal', 0)}
                            """
                            
                            st.download_button(
                                label="📊 Download Summary Report",
                                data=summary_report,
                                file_name=f"water_quality_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                mime="text/markdown"
                            )
                        
                    except Exception as e:
                        st.error(f"❌ Error processing predictions: {str(e)}")
                        with st.expander("Show error details"):
                            st.code(str(e))
                        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_data_exploration():
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📈 Advanced Data Analytics</h2>', unsafe_allow_html=True)
    
    data = load_sample_data()
    if data is None:
        return
    
    # Enhanced dataset overview
    st.markdown('<h3 class="section-header">📊 Dataset Intelligence</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card info-metric">
            <h3>📊 Samples</h3>
            <h2>{len(data):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card info-metric">
            <h3>🔢 Features</h3>
            <h2>{len(data.columns)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        countries = data['Country'].nunique() if 'Country' in data.columns else 0
        st.markdown(f"""
        <div class="metric-card info-metric">
            <h3>🌍 Countries</h3>
            <h2>{countries}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card warning-metric">
            <h3>❓ Missing</h3>
            <h2>{data.isnull().sum().sum():,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        memory_usage = data.memory_usage(deep=True).sum() / 1024 / 1024
        st.markdown(f"""
        <div class="metric-card info-metric">
            <h3>💾 Memory</h3>
            <h2>{memory_usage:.1f} MB</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Advanced visualizations
    create_advanced_plots(data)
    
    # Geographic analysis
    create_geographic_analysis(data)
    
    # Model performance
    show_model_performance()
    
    # Statistical insights
    st.markdown('<h3 class="section-header">📋 Statistical Summary</h3>', unsafe_allow_html=True)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    st.dataframe(
        data[numeric_cols].describe().style.background_gradient(axis=1),
        use_container_width=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Main title with animation
    st.markdown('<h1 class="big-title fade-in">🌊 Water Quality AI Analytics Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #a0aec0; margin-bottom: 3rem;">Advanced Machine Learning for Environmental Monitoring</p>', unsafe_allow_html=True)
    
    # Enhanced sidebar navigation
    st.sidebar.markdown('<h2 style="color: #63b3ed;">🧭 Navigation</h2>', unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Choose Analysis Mode",
        ["🏠 Dashboard", "🔍 Single Prediction", "📊 Batch Analytics", "📈 Data Intelligence", "🎯 Model Performance"]
    )
    
    # Add some sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📊 System Status
    🟢 AI Model: Active  
    🟢 Database: Connected  
    🟢 Analytics: Online  
    
    ### 💡 Quick Tips
    - Use **Single Prediction** for individual samples
    - Use **Batch Analytics** for multiple samples
    - Explore **Data Intelligence** for insights
    """)
    
    if page == "🏠 Dashboard":
        st.markdown("""
        <div class="fade-in">
        
        ## 🎯 Welcome to Advanced Water Quality Analytics
        
        This cutting-edge platform leverages **Artificial Intelligence** and **Machine Learning** to provide comprehensive water quality assessment and environmental monitoring.
        
        ### 🚀 Platform Capabilities
        
        **🔍 Individual Analysis**
        - Real-time water quality prediction
        - Detailed safety assessments
        - AI confidence scoring
        - Parameter optimization suggestions
        
        **📊 Batch Processing**
        - Large-scale dataset analysis
        - Statistical trend identification
        - Automated quality reporting
        - Export capabilities for further analysis
        
        **📈 Advanced Analytics**
        - Multi-dimensional data exploration
        - Correlation analysis with clustering
        - Geographic quality mapping
        - Temporal trend analysis
        - Outlier detection and insights
        
        **🎯 Model Intelligence**
        - Feature importance ranking
        - Performance metrics visualization
        - Prediction confidence analysis
        - Model interpretability tools
        
        ### 📋 Water Quality Parameters
        
        Our AI model analyzes **9 critical parameters**:
        
        | Parameter | Description | Impact |
        |-----------|-------------|---------|
        | **pH Level** | Acidity/Alkalinity (0-14) | 🔴 Critical |
        | **Dissolved Oxygen** | Oxygen content (mg/L) | 🔴 Critical |
        | **BOD** | Biochemical Oxygen Demand | 🟡 Important |
        | **Ammonia** | Nitrogen compound level | 🟡 Important |
        | **Temperature** | Water temperature (°C) | 🟢 Moderate |
        | **Orthophosphate** | Phosphorus content | 🟢 Moderate |
        | **Nitrogen** | Total nitrogen content | 🟡 Important |
        | **Nitrate** | Nitrate concentration | 🟡 Important |
        | **CCME Values** | Canadian quality index | 🔴 Critical |
        
        ### 🏆 Quality Classification System
        
        Our AI predicts water quality on a **5-tier system**:
        
        - 🟢 **Excellent**: Safe for all uses, pristine quality
        - 🔵 **Good**: High quality, suitable for most applications  
        - 🟡 **Fair**: Moderate quality, may need treatment
        - 🟠 **Marginal**: Poor quality, requires treatment
        - 🔴 **Poor**: Unsafe, significant treatment needed
        
        ---
        
        **🎓 Final Year Project** | **🏫 Academic Excellence** | **🌍 Environmental Impact**
        
        *Choose an analysis mode from the sidebar to begin your water quality assessment journey!*
        
        </div>
        """, unsafe_allow_html=True)
        
    elif page == "🔍 Single Prediction":
        show_single_prediction()
        
    elif page == "📊 Batch Analytics":
        show_batch_prediction()
        
    elif page == "📈 Data Intelligence":
        show_data_exploration()
        
    elif page == "🎯 Model Performance":
        show_model_performance()

if __name__ == "__main__":
    main()
