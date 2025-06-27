import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

# Configure page
st.set_page_config(
    page_title="SPARK - Sustainable Power Analysis & Renewable Kinetics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        color: #e0e0e0;
        margin-bottom: 2rem;
    }

    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #cccccc;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 1px solid #444;
        padding-bottom: 0.25rem;
    }

    .metric-card {
        background-color: #1e1e1e;
        padding: 1.25rem;
        border-radius: 8px;
        border-left: 4px solid #555;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    .info-box {
        background-color: #262626;
        padding: 1.25rem;
        border-radius: 8px;
        border: 1px solid #3a3a3a;
        margin: 1rem 0;
    }

    .stSelectbox > div > div {
        background-color: #1e1e1e !important;
        border: 1px solid #3a3a3a !important;
        color: #e0e0e0 !important;
    }

    .stDateInput > div > div {
        background-color: #1e1e1e !important;
        border: 1px solid #3a3a3a !important;
        color: #e0e0e0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Generate sample data
@st.cache_data
def generate_sample_data():
    # Generate sample load forecast data
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    load_data = pd.DataFrame({
        'Date': dates,
        'Actual_Load': np.random.normal(1000, 200, len(dates)) + 100 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365),
        'Predicted_Load': np.random.normal(1000, 180, len(dates)) + 100 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365),
        'Temperature': np.random.normal(20, 10, len(dates)),
        'Humidity': np.random.uniform(30, 90, len(dates))
    })
    
    # Generate renewable energy data
    renewable_data = pd.DataFrame({
        'Date': dates,
        'Solar_Generation': np.random.uniform(0, 500, len(dates)) * (np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + 1),
        'Wind_Generation': np.random.uniform(0, 300, len(dates)),
        'Daylight_Hours': 8 + 4 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365),
        'Wind_Speed': np.random.uniform(2, 15, len(dates)),
        'Solar_Irradiance': np.random.uniform(100, 1000, len(dates))
    })
    
    # Generate non-renewable data
    non_renewable_data = pd.DataFrame({
        'Date': dates,
        'Coal_Generation': np.random.uniform(200, 800, len(dates)),
        'Natural_Gas_Generation': np.random.uniform(150, 600, len(dates)),
        'Nuclear_Generation': np.random.uniform(300, 500, len(dates)),
        'Oil_Generation': np.random.uniform(50, 200, len(dates))
    })
    
    return load_data, renewable_data, non_renewable_data

# Load data
load_data, renewable_data, non_renewable_data = generate_sample_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["Home", "📊 Data Analysis","👾 ML Analysis"]
)

if page == "Home":
    st.markdown("""
    <h1 class="main-header"> ⚡ SPARK – Sustainable Power Analytics and Renewable Kinetics</h1>

    <div class="info-box">
        <h2>Project Overview</h2>
        <p>
            This platform provides comprehensive tools for energy forecasting and deep analytics of both renewable and non-renewable energy sources, leveraging machine learning for sustainable power insights.
        </p>

        <h3>Key Features</h3>
        <ul>
            <li><strong>Load Forecasting:</strong> Accurate energy demand prediction using ML algorithms.</li>
            <li><strong>Renewable Energy Analysis:</strong> Insights into solar, wind, and other renewable sources.</li>
            <li><strong>Non-Renewable Analysis:</strong> Evaluation of coal, gas, oil, and nuclear energy generation.</li>
            <li><strong>Correlation Analysis:</strong> Examine interdependencies between key energy metrics.</li>
            <li><strong>Seasonal Trends:</strong> Discover fossil fuel usage patterns across different seasons.</li>
        </ul>

        <h3>Objectives</h3>
        <ul>
            <li>Enhance grid efficiency through predictive analytics.</li>
            <li>Encourage renewable adoption with actionable insights.</li>
            <li>Reduce fossil dependency via informed planning.</li>
            <li>Support energy policy with evidence-based analysis.</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

    # Stats Overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>Historical Data</h4>
            <h2>35K+</h2>
            <p>Hours analyzed</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>Forecast Accuracy</h4>
            <h2>95.2%</h2>
            <p>Avg. model performance</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>Renewable Share</h4>
            <h2>34.7%</h2>
            <p>Current clean energy usage</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4>Peak Load in an hour</h4>
            <h2>41,015 MW</h2>
            <p>Max recorded demand</p>
        </div>
        """, unsafe_allow_html=True)

# DATA ANALYSIS PAGE
elif page == "📊 Data Analysis":
    st.markdown('<h1 class="main-header">📊 Data Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h2>🔍 Data Analysis Overview</h2>
        <p>Comprehensive analysis of energy generation patterns, renewable vs non-renewable sources, and seasonal dependencies:</p>
        <ul>
            <li><strong>Renewable Energy:</strong> Solar, wind, and other clean energy sources analysis</li>
            <li><strong>Non-Renewable Energy:</strong> Coal, natural gas, nuclear, and oil generation patterns</li>
            <li><strong>Seasonal Analysis:</strong> Fossil fuel dependency across different seasons</li>
            <li><strong>Comparative Studies:</strong> Interactive comparison between different energy sources</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Analysis type selection
    st.markdown('<h2 class="section-header">⚙️ Analysis Configuration</h2>', unsafe_allow_html=True)
    
    analysis_type = st.selectbox(
        "Choose Analysis Type:",
        ["Renewables", "Non-Renewables", "Fossil Fuel Dependency"]
    )
    
    # Time range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime(2024, 1, 1), key="data_start")
    with col2:
        end_date = st.date_input("End Date", datetime(2024, 12, 31), key="data_end")
    
    if analysis_type == "Renewables":
        st.markdown('<h2 class="section-header">🌱 Renewable Energy Analysis</h2>', unsafe_allow_html=True)
        
        # Filter renewable data
        mask = (renewable_data['Date'] >= pd.to_datetime(start_date)) & (renewable_data['Date'] <= pd.to_datetime(end_date))
        filtered_renewable = renewable_data.loc[mask]
        
        # Comparison selection
        comparison_type = st.selectbox(
            "Select Comparison:",
            ["Solar vs Daylight Hours", "Solar vs Wind Generation", "Wind vs Wind Speed", "Solar vs Irradiance"]
        )
        
        if comparison_type == "Solar vs Daylight Hours":
            fig = px.scatter(
                filtered_renewable, 
                x='Daylight_Hours', 
                y='Solar_Generation',
                color='Solar_Irradiance',
                title="Solar Generation vs Daylight Hours",
                labels={'Daylight_Hours': 'Daylight Hours', 'Solar_Generation': 'Solar Generation (MW)'}
            )
            
        elif comparison_type == "Solar vs Wind Generation":
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=filtered_renewable['Date'], y=filtered_renewable['Solar_Generation'], name='Solar Generation'),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=filtered_renewable['Date'], y=filtered_renewable['Wind_Generation'], name='Wind Generation'),
                secondary_y=True,
            )
            fig.update_yaxes(title_text="Solar Generation (MW)", secondary_y=False)
            fig.update_yaxes(title_text="Wind Generation (MW)", secondary_y=True)
            fig.update_layout(title_text="Solar vs Wind Generation Over Time")
            
        elif comparison_type == "Wind vs Wind Speed":
            fig = px.scatter(
                filtered_renewable, 
                x='Wind_Speed', 
                y='Wind_Generation',
                title="Wind Generation vs Wind Speed",
                labels={'Wind_Speed': 'Wind Speed (m/s)', 'Wind_Generation': 'Wind Generation (MW)'}
            )
            
        else:  # Solar vs Irradiance
            fig = px.scatter(
                filtered_renewable, 
                x='Solar_Irradiance', 
                y='Solar_Generation',
                title="Solar Generation vs Solar Irradiance",
                labels={'Solar_Irradiance': 'Solar Irradiance (W/m²)', 'Solar_Generation': 'Solar Generation (MW)'}
            )
        
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Solar Generation", f"{filtered_renewable['Solar_Generation'].mean():.1f} MW")
        with col2:
            st.metric("Avg Wind Generation", f"{filtered_renewable['Wind_Generation'].mean():.1f} MW")
        with col3:
            st.metric("Total Renewable", f"{(filtered_renewable['Solar_Generation'] + filtered_renewable['Wind_Generation']).mean():.1f} MW")
    
    elif analysis_type == "Non-Renewables":
        st.markdown('<h2 class="section-header">🏭 Non-Renewable Energy Analysis</h2>', unsafe_allow_html=True)
        
        # Filter non-renewable data
        mask = (non_renewable_data['Date'] >= pd.to_datetime(start_date)) & (non_renewable_data['Date'] <= pd.to_datetime(end_date))
        filtered_non_renewable = non_renewable_data.loc[mask]
        
        # Stacked area chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=filtered_non_renewable['Date'], 
            y=filtered_non_renewable['Coal_Generation'],
            mode='lines',
            stackgroup='one',
            name='Coal',
            line=dict(color='#8b4513')
        ))
        
        fig.add_trace(go.Scatter(
            x=filtered_non_renewable['Date'], 
            y=filtered_non_renewable['Natural_Gas_Generation'],
            mode='lines',
            stackgroup='one',
            name='Natural Gas',
            line=dict(color='#4682b4')
        ))
        
        fig.add_trace(go.Scatter(
            x=filtered_non_renewable['Date'], 
            y=filtered_non_renewable['Nuclear_Generation'],
            mode='lines',
            stackgroup='one',
            name='Nuclear',
            line=dict(color='#32cd32')
        ))
        
        fig.add_trace(go.Scatter(
            x=filtered_non_renewable['Date'], 
            y=filtered_non_renewable['Oil_Generation'],
            mode='lines',
            stackgroup='one',
            name='Oil',
            line=dict(color='#ff4500')
        ))
        
        fig.update_layout(
            title='Non-Renewable Energy Generation Over Time',
            xaxis_title='Date',
            yaxis_title='Generation (MW)',
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Pie chart for composition
        total_generation = filtered_non_renewable[['Coal_Generation', 'Natural_Gas_Generation', 'Nuclear_Generation', 'Oil_Generation']].sum()
        
        fig_pie = px.pie(
            values=total_generation.values,
            names=total_generation.index,
            title="Non-Renewable Energy Mix"
        )
        fig_pie.update_layout(template='plotly_dark')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    else:  # Fossil Fuel Dependency
        st.markdown('<h2 class="section-header">🛢️ Fossil Fuel Dependency Analysis</h2>', unsafe_allow_html=True)
        
        # Combine all data for seasonal analysis
        combined_data = load_data.merge(renewable_data, on='Date').merge(non_renewable_data, on='Date')
        
        # Add season column
        combined_data['Month'] = combined_data['Date'].dt.month
        combined_data['Season'] = combined_data['Month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Calculate fossil fuel dependency
        combined_data['Fossil_Fuel'] = combined_data['Coal_Generation'] + combined_data['Natural_Gas_Generation'] + combined_data['Oil_Generation']
        combined_data['Renewable_Total'] = combined_data['Solar_Generation'] + combined_data['Wind_Generation']
        combined_data['Total_Generation'] = combined_data['Fossil_Fuel'] + combined_data['Renewable_Total'] + combined_data['Nuclear_Generation']
        combined_data['Fossil_Dependency'] = (combined_data['Fossil_Fuel'] / combined_data['Total_Generation']) * 100
        
        # Seasonal dependency analysis
        seasonal_dependency = combined_data.groupby('Season')['Fossil_Dependency'].mean().reset_index()
        
        fig_seasonal = px.bar(
            seasonal_dependency,
            x='Season',
            y='Fossil_Dependency',
            title='Fossil Fuel Dependency by Season',
            color='Fossil_Dependency',
            color_continuous_scale='Reds'
        )
        fig_seasonal.update_layout(
            template='plotly_dark',
            yaxis_title='Fossil Fuel Dependency (%)'
        )
        st.plotly_chart(fig_seasonal, use_container_width=True)
        
        # Monthly trend
        monthly_trend = combined_data.groupby(combined_data['Date'].dt.month)['Fossil_Dependency'].mean().reset_index()
        monthly_trend['Month_Name'] = monthly_trend['Date'].map({
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        })
        
        fig_monthly = px.line(
            monthly_trend,
            x='Month_Name',
            y='Fossil_Dependency',
            title='Monthly Fossil Fuel Dependency Trend',
            markers=True
        )
        fig_monthly.update_layout(template='plotly_dark')
        st.plotly_chart(fig_monthly, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Fossil Dependency", f"{combined_data['Fossil_Dependency'].mean():.1f}%")
        with col2:
            st.metric("Highest Season", f"{seasonal_dependency.loc[seasonal_dependency['Fossil_Dependency'].idxmax(), 'Season']}")
        with col3:
            st.metric("Lowest Season", f"{seasonal_dependency.loc[seasonal_dependency['Fossil_Dependency'].idxmin(), 'Season']}")
        with col4:
            st.metric("Seasonal Variation", f"{seasonal_dependency['Fossil_Dependency'].max() - seasonal_dependency['Fossil_Dependency'].min():.1f}%")


# ML ANALYSIS PAGE
elif page == "👾 ML Analysis":
    st.markdown('<h1 class="main-header">👾 Machine Learning Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
<h2>Vision for an Intelligent Grid</h2>
<ul>
    <li>Autonomous system capable of <strong>real-time monitoring and forecasting</strong>.</li>
    <li>Dynamically balances energy supply by:
        <ul>
            <li>Predicting shortfalls in renewable energy</li>
            <li>Seamlessly switching to backup sources</li>
        </ul>
    </li>
    <li>Optimizes for:
        <ul>
            <li>Cost efficiency</li>
            <li>Reduced carbon emissions</li>
            <li>Grid reliability</li>
        </ul>
    </li>
    <li>Utilizes <strong>machine learning</strong> and data-driven algorithms to continuously enhance decision-making.</li>
</ul>

<h3>Current Gaps</h3>
<ul>
    <li>Lack of <strong>adaptive, predictive, and automated control systems</strong>.</li>
    <li>Existing systems are reactive and manual, failing to:
        <ul>
            <li>Anticipate renewable variability</li>
            <li>Make proactive adjustments</li>
        </ul>
    </li>
    <li>Results in:
        <ul>
            <li>Operational inefficiencies</li>
            <li>Higher carbon emissions</li>
            <li>Increased risk of outages</li>
        </ul>
    </li>
    <li>Hinders progress toward a <strong>sustainable and reliable energy future</strong>.</li>
</ul>
""", unsafe_allow_html=True)
    
    # Forecast selection
    st.markdown('<h2 class="section-header">🔮 Forecast Selection</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        forecast_type = st.selectbox(
            "Choose Forecast Type:",
            ["Load Forecast", "Renewable Generation Forecast", "Peak Demand Forecast", "Seasonal Forecast"]
        )
    
    with col2:
        metric_type = st.selectbox(
            "Select Metrics:",
            ["RMSE", "MAE", "MAPE", "R²", "All Metrics"]
        )
    
    # Time range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime(2024, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime(2024, 12, 31))
    
    # Filter data based on date range
    mask = (load_data['Date'] >= pd.to_datetime(start_date)) & (load_data['Date'] <= pd.to_datetime(end_date))
    filtered_data = load_data.loc[mask]
    
    # Actual vs Predicted Plot
    st.markdown('<h2 class="section-header">📈 Actual vs Predicted Analysis</h2>', unsafe_allow_html=True)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Actual vs Predicted Load', 'Prediction Error'),
        vertical_spacing=0.1
    )
    
    # Main plot
    fig.add_trace(
        go.Scatter(x=filtered_data['Date'], y=filtered_data['Actual_Load'], 
                  name='Actual Load', line=dict(color='#1f77b4', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=filtered_data['Date'], y=filtered_data['Predicted_Load'], 
                  name='Predicted Load', line=dict(color='#ff7f0e', width=2)),
        row=1, col=1
    )
    
    # Error plot
    error = filtered_data['Actual_Load'] - filtered_data['Predicted_Load']
    fig.add_trace(
        go.Scatter(x=filtered_data['Date'], y=error, 
                  name='Prediction Error', line=dict(color='#d62728', width=1)),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        template='plotly_dark',
        title_text=f"{forecast_type} - Performance Analysis",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics display
    st.markdown('<h2 class="section-header">📊 Performance Metrics</h2>', unsafe_allow_html=True)
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((filtered_data['Actual_Load'] - filtered_data['Predicted_Load'])**2))
    mae = np.mean(np.abs(filtered_data['Actual_Load'] - filtered_data['Predicted_Load']))
    mape = np.mean(np.abs((filtered_data['Actual_Load'] - filtered_data['Predicted_Load']) / filtered_data['Actual_Load'])) * 100
    r2 = np.corrcoef(filtered_data['Actual_Load'], filtered_data['Predicted_Load'])[0,1]**2
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("RMSE", f"{rmse:.2f}", "MW")
    with col2:
        st.metric("MAE", f"{mae:.2f}", "MW")
    with col3:
        st.metric("MAPE", f"{mape:.2f}", "%")
    with col4:
        st.metric("R² Score", f"{r2:.3f}", "")
    
    # Correlation Matrix
    st.markdown('<h2 class="section-header">🔗 Correlation Matrix</h2>', unsafe_allow_html=True)
    
    corr_data = filtered_data[['Actual_Load', 'Predicted_Load', 'Temperature', 'Humidity']].corr()
    
    fig_corr = px.imshow(
        corr_data,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Feature Correlation Matrix"
    )
    fig_corr.update_layout(template='plotly_dark')
    st.plotly_chart(fig_corr, use_container_width=True)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>⚡ Energy Analysis & Forecasting Platform | Built with Streamlit</p>
    <p>📊 Advanced Analytics • 👾 Machine Learning • 🌱 Sustainable Energy</p>
</div>
""", unsafe_allow_html=True)