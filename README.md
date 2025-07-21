# ⚡ SPARK — Sustainable Power Analytics and Renewable Kinetics

**Big Data Energy Analytics with Hadoop MapReduce and Apache Spark ML**

---

## 📌 Overview

**SPARK** is a scalable, modular, and extensible energy analytics platform designed for processing large-scale renewable energy datasets using the Hadoop MapReduce framework. It offers data analytics, machine learning-based forecasting, and energy trend insights in a fully modular setup.

---
Our energy analytics platform is designed to achieve the following key objectives:

1. **Enhance Grid Efficiency through Predictive Analytics**  
   Leverage machine learning and historical data to forecast energy demand, optimize grid operations, and minimize load imbalance.

2. **Encourage Renewable Adoption with Actionable Insights**  
   Provide clear, data-driven analysis of solar, wind, and other renewable sources to empower stakeholders in making sustainable energy decisions.

3. **Reduce Fossil Dependency via Informed Planning**  
   Analyze usage patterns to support strategies that reduce reliance on coal, oil, and gas, promoting a cleaner energy mix.

4. **Support Energy Policy with Evidence-Based Analysis**  
   Deliver insights derived from historical data to assist policymakers in crafting effective, climate-resilient energy policies.
---

## 🔧 Features

- ⚡ Hadoop MapReduce-based big data processing
- 📊 Machine learning models for forecasting and anomaly detection
- 🧑‍💻 UI for interactive visualization and control
- 🔗 Support for multi-source datasets (solar, wind, Fossils)

---
## 🔑 What's SPARK doing ?

### 🔮 Load Forecasting
- Accurate energy demand prediction using advanced machine learning algorithms.
- Enables better planning for energy distribution and grid stability.
- Supports short-term and long-term forecasting.

### ☀️ Renewable Energy Analysis
- In-depth insights into solar, wind, and other renewable energy sources.
- Tracks generation patterns, efficiency, and capacity factors.
- Supports evaluation of environmental and economic impacts.

### ⚡ Non-Renewable Energy Analysis
- Evaluation of coal, gas, oil, and nuclear energy generation.
- Assesses output efficiency and carbon emissions.
- Provides a comparative view with renewable sources.

### 🔗 Correlation Analysis
- Analyze interdependencies between key energy metrics such as:
  - Demand vs. Generation
  - Renewable vs. Non-renewable output
  - Weather vs. Energy production
- Identify hidden patterns and potential optimization strategies.

### 🌦️ Seasonal Trends
- Discover fossil fuel usage patterns across different seasons.
- Detect seasonal dependencies in energy consumption and production.
- Supports energy policy planning and climate impact analysis.

---
# 📊 Data Analysis Overview

A comprehensive analysis of energy generation patterns, renewable vs non-renewable sources, and seasonal dependencies to support data-driven decision making.

## 🌱 Renewable Energy
- Analysis of solar, wind, and other clean energy sources.
- Tracks generation trends, efficiency, and adoption rates.
- Assesses the contribution of renewables to total energy supply.

## 🔥 Non-Renewable Energy
- Detailed evaluation of coal, natural gas, nuclear, and oil generation patterns.
- Examines output variability, cost-efficiency, and environmental impact.
- Provides insights into fossil fuel reliance and transition potential.

## ❄️ Seasonal Analysis
- Identifies fossil fuel dependency across different seasons.
- Reveals peak consumption periods and seasonal generation trends.
- Supports strategic energy storage and policy planning.

## ⚖️ Comparative Studies
- Interactive comparison between renewable and non-renewable sources.
- Visual analysis of generation capacity, cost, emissions, and usage.
- Helps stakeholders evaluate trade-offs and make informed energy choices.

---
## 📁 Project Structure

```bash
SPARK-Sustainable-Power-Analytics-and-Renewable-Kinetics/
│
├── data/              # datasets
├── mapreduce/         # Java MapReduce jobs for large-scale data processing
│   ├── FossilFuelDependency/           # Job to calculate fossil fuel dependency
│   │   ├── ffd.csv                     # Output Dataset
│   │   ├── FossilFuelDependencyDriver.java
│   │   ├── FossilFuelDependencyMapper.java
│   │   ├── FossilFuelDependencyReducer.java
│   │   └── readme                      # Notes/documentation for this module
│   ├── LoadForecast/                   # Energy demand forecasting
│   ├── renewable/                      # Renewable source processing 
│   ├── nonrenewable/                   # Coal, gas, etc. analytics
│   ├── weather/                        # Impact of temperature, wind, etc.
├── ml-modelling/      # Jupyter Notebooks and ML models for energy forecasting
│   ├── Models           # Joblin Files of models
│   ├── renewable-predictions.ipynb     #ipynb file for pre processing and training
├── monitoring/        # Scripts and modules for real-time monitoring
├── ui/                # Streamlit/CLI-based UI interface
│   ├── app.py         # Without System monitoring
│   ├── testui.py      # With System monitoring
│   ├── testapp.py     # Test UI for only System monitoring
├── commands.txt       # Execution and helper commands
├── requirements.txt   # Python dependencies
└── README.md          # You're here!
```
# 🚀 Tech Stack & Workflow Summary

This document outlines the architecture of a big data processing environment containerized with Docker and details the command-line workflow for extracting processed data.

---

## 🛠️ Tech Stack Overview

The environment is built on **Docker**, which isolates and manages two primary service stacks. This separation allows for **modularity** and **independent scaling** of the processing and analysis layers.

---

## 📦 Containerization & Orchestration

- **Docker Desktop**: Core platform for running and managing the containerized applications on a local machine.
- **Docker Compose**: Defines and orchestrates multi-container services. The setup consists of two distinct stacks:
  - **Hadoop stack**: For distributed storage and batch processing.
  - **Spark/Jupyter stack**: For interactive data analysis.

---

## 🐘 Big Data Frameworks

### Apache Hadoop Stack (`docker-hadoop`)
- **HDFS (Hadoop Distributed File System)**:  
  Managed by the `namenode` and `datanode` containers. Provides resilient, distributed storage for large datasets.
- **YARN (Yet Another Resource Negotiator)**:  
  Managed by the `resourcemanager` and `nodemanager` containers. Handles job scheduling and resource management.
- **MapReduce**:  
  The primary framework for parallel processing of data stored in HDFS. Output files typically follow the `part-r-00000` naming convention.

### Apache Spark Stack (`jupyter-spark`)
- **Apache Spark**:  
  Fast, in-memory data processing engine running in its own container. Interacts with HDFS for advanced analytics.
- **Jupyter Notebook**:  
  Web-based interface for writing and executing Python/PySpark code, ideal for exploratory data analysis.

---

## ⚙️ Data Processing Workflow

The shell commands provided below demonstrate a repeatable process for extracting results of MapReduce jobs and preparing them for local analysis.

---

### Step-by-Step Data Extraction

The workflow is executed for three datasets:
- **Renewable Energy (rde)**
- **Non-Renewable Energy (nrde)**
- **Weekly Load Actuals (wla)**

#### 1. Execute MapReduce Job *(Implicit Step)*
A MapReduce job processes raw data and outputs results to HDFS directories:
/labelout/rde/,
/labelout/nrde/,
/labelout/wla/.
#### 2. Extract Raw Output from HDFS
Use the `hdfs dfs -cat` command to read the `part-r-00000` file and write to a temporary local file inside the `namenode` container:

```bash
# Example for renewable energy data
hdfs dfs -cat /labelout/rde/part-r-00000 > rde_raw.csv
```
#### 3. Prepend CSV Header

Add a header row using `echo` and `cat`, creating a clean, final CSV file:

```bash
# Example for renewable energy data
echo "time,generation biomass,..." | cat - rde_raw.csv > rme.csv
```
#### 4. Copy Final CSV to Host Machine
Transfer the final CSV from the namenode container to the local Windows filesystem:

```bash
# Example for renewable energy data
docker cp namenode:/root/renewable/monthlyenergy/rme.csv D:\VI\BDT\LABEL\mapreduce\renewable\monthlyenergy
```
#### 5. ML Modelling and Analysis
Refer to `ml-modelling` folder where models are built and saved in .joblibs format under `ml-modelling/model`.


---
### How to run
Since all the Mapreduce jobs have already been executed and final files stored, onyl part left out is the UI. To run it,
```bash
streamlit run ui/testui.py
```

Make sure to have [Open Hardware Monitor](https://openhardwaremonitor.org/) running in background with its remote web server enabled to get System Monitoring. If system monitoring isn't needed, run,
```bash
streamlit run ui/app.py
```


