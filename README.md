# ⚡ SPARK — Sustainable Power Analytics and Renewable Kinetics

**Big Data Energy Analytics with Hadoop MapReduce and Apache Spark ML**

---

## 📌 Overview

**SPARK** is a scalable, modular, and extensible energy analytics platform designed for processing large-scale renewable energy datasets using the Hadoop MapReduce framework. It offers data analytics, machine learning-based forecasting, and energy trend insights in a fully modular setup.

---

## 🔧 Features

- ⚡ Hadoop MapReduce-based big data processing
- 📊 Machine learning models for forecasting and anomaly detection
- 🧑‍💻 UI for interactive visualization and control
- 🔗 Support for multi-source datasets (solar, wind, Fossils)

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
