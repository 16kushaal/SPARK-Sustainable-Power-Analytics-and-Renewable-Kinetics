import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import requests
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="SPARK - Sustainable Power Analysis & Renewable Kinetics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Modern UI Styling ---
st.markdown("""
<style>
    /* General Styles */
    .main-header {
        font-size: 2.5rem; font-weight: 600; text-align: center;
        color: #e0e0e0; margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem; font-weight: 600; color: #cccccc;
        margin: 1.5rem 0 1rem 0; border-bottom: 1px solid #444;
        padding-bottom: 0.25rem;
    }
    .info-box {
        background-color: #262626; padding: 1.25rem; border-radius: 8px;
        border: 1px solid #3a3a3a; margin: 1rem 0;
    }
    .info-box h2, .info-box h3 { color: #e0e0e0; }
    .info-box p, .info-box ul, .info-box li { color: #cccccc; line-height: 1.6; }

    /* System Monitor Specific Styles */
    .monitor-card {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #3a3a3a;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .monitor-card-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #e0e0e0;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #444;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0.2rem;
        border-bottom: 1px solid #2a2a2a;
    }
    .metric-row:last-child {
        border-bottom: none;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #cccccc;
    }
    .metric-value {
        font-size: 1.0rem;
        font-weight: 500;
        color: #e0e0e0;
        background-color: #2a3f54;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
    }
    .connection-error {
        background-color: #ff4444; color: white; padding: 1rem;
        border-radius: 8px; text-align: left;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- OpenHardwareMonitor Integration Class ---
class OHMWebMonitor:
    def __init__(self, host='localhost', port=8085):
        self.data_url = f'http://{host}:{port}/data.json'
        self.connected = False

    def get_sensor_data(self):
        try:
            response = requests.get(self.data_url, timeout=2)
            if response.status_code == 200:
                self.connected = True
                return response.json()
            self.connected = False
            return None
        except requests.exceptions.RequestException:
            self.connected = False
            return None

    def parse_sensors(self, data):
        sensors = {
            'cpu': {'temperature': 0, 'load': 0, 'power': 0, 'name': 'CPU'},
            'gpu': {'temperature': 0, 'load': 0, 'power': 0, 'name': 'GPU'},
            'ram': {'load': 0, 'used': 0, 'available': 0, 'total': 0},
            'storage': {'temperature': 0, 'used_space': 0, 'name': 'Storage'}
        }
        if not data: return sensors

        def extract_number(value_str):
            if not value_str: return 0.0
            match = re.search(r'([\d.]+)', str(value_str))
            return float(match.group(1)) if match else 0.0

        def find_node(node, text_to_find, search_image_url=False):
            # Check current node's text or ImageURL
            if text_to_find in node.get('Text', '').lower():
                return node
            if search_image_url and text_to_find in node.get('ImageURL', '').lower():
                return node
            # Recurse into children
            for child in node.get('Children', []):
                found = find_node(child, text_to_find, search_image_url)
                if found: return found
            return None

        def get_sensor_value(hardware_node, sensor_group, sensor_name):
            if not hardware_node: return 0
            group_node = find_node(hardware_node, sensor_group)
            if not group_node: return 0
            sensor_node = find_node(group_node, sensor_name)
            if sensor_node:
                return extract_number(sensor_node.get('Value'))
            return 0

        # Find main hardware nodes by their icon URL, which is more reliable
        cpu_node = find_node(data, 'cpu', search_image_url=True)
        ram_node = find_node(data, 'ram', search_image_url=True)
        storage_node = find_node(data, 'hdd', search_image_url=True) # The hdd icon is often used for SSDs too
        gpu_node = find_node(data, 'gpu', search_image_url=True)

        # Parse CPU
        if cpu_node:
            sensors['cpu']['name'] = cpu_node.get('Text', 'CPU')
            sensors['cpu']['load'] = get_sensor_value(cpu_node, 'load', 'cpu total')
            sensors['cpu']['temperature'] = get_sensor_value(cpu_node, 'temperatures', 'cpu package')
            sensors['cpu']['power'] = get_sensor_value(cpu_node, 'powers', 'cpu package')

        # Parse RAM
        if ram_node:
            sensors['ram']['load'] = get_sensor_value(ram_node, 'load', 'memory')
            sensors['ram']['used'] = get_sensor_value(ram_node, 'data', 'used memory')
            sensors['ram']['available'] = get_sensor_value(ram_node, 'data', 'available memory')
            if sensors['ram']['used'] > 0 and sensors['ram']['available'] > 0:
                sensors['ram']['total'] = sensors['ram']['used'] + sensors['ram']['available']

        # Parse Storage
        if storage_node:
            sensors['storage']['name'] = storage_node.get('Text', 'Storage')
            sensors['storage']['temperature'] = get_sensor_value(storage_node, 'temperatures', 'temperature')
            sensors['storage']['used_space'] = get_sensor_value(storage_node, 'load', 'used space')

        # Parse GPU
        if gpu_node:
            sensors['gpu']['name'] = gpu_node.get('Text', 'GPU')
            sensors['gpu']['load'] = get_sensor_value(gpu_node, 'load', 'gpu core')
            sensors['gpu']['temperature'] = get_sensor_value(gpu_node, 'temperatures', 'gpu core')
            sensors['gpu']['power'] = get_sensor_value(gpu_node, 'powers', 'gpu total')
            
        return sensors


# --- Data Loading (Cached) ---
@st.cache_data
def load_energy_data():
    try:
        daily_renewable = pd.read_csv('./mapreduce/renewable/dailyenergy/daily.csv')
        daily_renewable['Date'] = pd.to_datetime(daily_renewable['time'], format='%d-%m-%y')
        return {'daily_renewable': daily_renewable}
    except FileNotFoundError:
        return None

# --- UI Functions ---
def create_gauge(value, title, max_val=100, suffix='%', color='#00b894'):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16, 'color': '#cccccc'}},
        gauge={
            'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0.2)",
            'borderwidth': 1,
            'bordercolor': "#444",
        },
        number={'suffix': suffix, 'font': {'size': 28, 'color': '#e0e0e0'}}
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def display_system_monitor():
    st.markdown('<h1 class="main-header">🖥️ System Hardware Monitor</h1>', unsafe_allow_html=True)

    # Initialize session state for auto-refresh and last update time
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    if 'last_refresh_time' not in st.session_state:
        st.session_state.last_refresh_time = time.time()

    monitor = OHMWebMonitor()
    
    # --- Control Panel ---
    st.markdown("---")
    cols = st.columns([2, 2, 1, 1])
    
    last_refreshed_str = datetime.fromtimestamp(st.session_state.last_refresh_time).strftime('%H:%M:%S')
    cols[0].markdown(f"🕒 **Last Update:** `{last_refreshed_str}`")

    with cols[1]:
        if st.session_state.auto_refresh:
            st.success("🟢 Live Refresh is ON")
        else:
            st.warning("🟡 Live Refresh is OFF")
    with cols[2]:
        if st.button("🔄 Refresh"):
            st.session_state.last_refresh_time = time.time()
            st.rerun()
    with cols[3]:
        st.session_state.auto_refresh = st.toggle("Live", value=st.session_state.get('auto_refresh', True), key="auto_refresh_toggle")
    st.markdown("---")


    data = monitor.get_sensor_data()

    if not monitor.connected or not data:
        st.markdown("""
        <div class="connection-error">
            <strong>❌ Connection Failed: Could not connect to Open Hardware Monitor.</strong>
            <br><br>
            <strong>Troubleshooting Steps:</strong>
            <ol>
                <li><strong>Download and Install:</strong> <a href="https://openhardwaremonitor.org/downloads/" target="_blank" style="color: white;">Open Hardware Monitor</a></li>
                <li><strong>Run as Administrator:</strong> Right-click the .exe and select "Run as administrator".</li>
                <li><strong>Enable Web Server:</strong> In the app, go to Options → Remote Web Server → Check "Run".</li>
                <li><strong>Check Firewall:</strong> Ensure your firewall is not blocking the connection on port 8085.</li>
                <li><strong>Test URL:</strong> Visit <a href="http://localhost:8085/data.json" target="_blank" style="color: white;">http://localhost:8085/data.json</a> in your browser. You should see a block of text.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        return

    sensors = monitor.parse_sensors(data)

    # --- Main Gauges ---
    cols = st.columns(4)
    with cols[0]:
        st.plotly_chart(create_gauge(sensors['cpu']['load'], "CPU Load", color='#6c5ce7'), use_container_width=True)
    with cols[1]:
        st.plotly_chart(create_gauge(sensors['cpu']['temperature'], "CPU Temp", suffix='°C', color='#fd79a8'), use_container_width=True)
    with cols[2]:
        st.plotly_chart(create_gauge(sensors['ram']['load'], "RAM Usage", color='#00cec9'), use_container_width=True)
    with cols[3]:
        st.plotly_chart(create_gauge(sensors['storage']['used_space'], "Disk Usage", color='#fab1a0'), use_container_width=True)

    # --- Detailed Cards ---
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(4)

    with cols[0]:
        st.markdown(f"""
        <div class="monitor-card">
            <div class="monitor-card-header">🔥 {sensors["cpu"]["name"]}</div>
            <div class="metric-row"><span class="metric-label">🌡️ Temperature</span> <span class="metric-value">{sensors['cpu']['temperature']:.1f} °C</span></div>
            <div class="metric-row"><span class="metric-label">📊 Load</span> <span class="metric-value">{sensors['cpu']['load']:.1f} %</span></div>
            <div class="metric-row"><span class="metric-label">⚡ Power</span> <span class="metric-value">{sensors['cpu']['power']:.1f} W</span></div>
        </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        st.markdown(f"""
        <div class="monitor-card">
            <div class="monitor-card-header">💾 Memory (RAM)</div>
            <div class="metric-row"><span class="metric-label">📊 Usage</span> <span class="metric-value">{sensors['ram']['load']:.1f} %</span></div>
            <div class="metric-row"><span class="metric-label">💿 Used</span> <span class="metric-value">{sensors['ram']['used']:.1f} GB</span></div>
            <div class="metric-row"><span class="metric-label">🗃️ Total</span> <span class="metric-value">{sensors['ram']['total']:.1f} GB</span></div>
        </div>
        """, unsafe_allow_html=True)

    with cols[2]:
        st.markdown(f"""
        <div class="monitor-card">
            <div class="monitor-card-header">💿 {sensors["storage"]["name"]}</div>
            <div class="metric-row"><span class="metric-label">🌡️ Temperature</span> <span class="metric-value">{sensors['storage']['temperature']:.1f} °C</span></div>
            <div class="metric-row"><span class="metric-label">📊 Used Space</span> <span class="metric-value">{sensors['storage']['used_space']:.1f} %</span></div>
        </div>
        """, unsafe_allow_html=True)

    with cols[3]:
        if sensors['gpu']['load'] > 0 or sensors['gpu']['temperature'] > 0:
            st.markdown(f"""
            <div class="monitor-card">
                <div class="monitor-card-header">🎮 {sensors["gpu"]["name"]}</div>
                <div class="metric-row"><span class="metric-label">🌡️ Temperature</span> <span class="metric-value">{sensors['gpu']['temperature']:.1f} °C</span></div>
                <div class="metric-row"><span class="metric-label">📊 Load</span> <span class="metric-value">{sensors['gpu']['load']:.1f} %</span></div>
                <div class="metric-row"><span class="metric-label">⚡ Power</span> <span class="metric-value">{sensors['gpu']['power']:.1f} W</span></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="monitor-card">
                <div class="monitor-card-header">🎮 GPU</div>
                <div style="text-align: center; padding-top: 2rem; color: #666;">
                    No compatible GPU detected.
                </div>
            </div>
            """, unsafe_allow_html=True)


    # Auto-refresh logic
    if st.session_state.auto_refresh:
        time.sleep(5)
        st.session_state.last_refresh_time = time.time()
        st.rerun()

# --- Main App Logic ---
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Home", "📊 Data Analysis", "👾 ML Analysis", "🖥️ System Monitor"]
    )

    if page == "Home":
        st.markdown('<h1 class="main-header">⚡ SPARK – Sustainable Power Analytics and Renewable Kinetics</h1>', unsafe_allow_html=True)
        st.info("Navigate to other sections using the sidebar.")

    elif page == "📊 Data Analysis":
        st.markdown('<h1 class="main-header">📊 Data Analysis Dashboard</h1>', unsafe_allow_html=True)
        energy_data = load_energy_data()
        if energy_data:
            st.success("Energy data loaded successfully.")
        else:
            st.error("Could not load energy data files.")

    elif page == "👾 ML Analysis":
        st.markdown('<h1 class="main-header">👾 Machine Learning Analysis</h1>', unsafe_allow_html=True)
        st.info("This section contains machine learning forecast models.")

    elif page == "🖥️ System Monitor":
        display_system_monitor()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>⚡ Energy Analysis & Forecasting Platform | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
