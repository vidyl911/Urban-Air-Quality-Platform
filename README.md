# Urban Air Quality & Spatiotemporal Intelligence Platform

An end-to-end data and machine learning platform for **spatiotemporal PM2.5 forecasting** using environmental observations, weather conditions, and spatial context.

The project combines distributed data processing, spatial analytics, streaming, workflow orchestration, deep learning, experiment tracking, and cloud infrastructure into a production-oriented ML pipeline.

## Architecture

```text
Data Sources
    │
    ├── Air Quality
    ├── Weather
    └── Spatial Context
    │
    ▼
Ingestion
    │
    ▼
Bronze Layer
    │
    ▼
Silver Layer
    │
    ├── Temporal Alignment
    ├── Data Quality
    └── Spatial Processing
    │
    ▼
Gold Feature Layer
    │
    ▼
PM2.5 Time-Series Sequences
    │
    ▼
RNN / LSTM / GRU
    │
    ├── Evaluation
    ├── MLflow
    └── Model Registry
    │
    ▼
Predictions
```

## Technology Stack

### Data Engineering

* Python
* SQL
* PySpark
* Apache Kafka
* Apache Airflow

### Spatial Data

* PostGIS
* GeoPandas
* Shapely
* PyProj

### Machine Learning

* PyTorch
* RNN
* LSTM
* GRU
* Scikit-learn

### MLOps & Data Management

* MLflow
* DataHub
* Pytest
* Ruff

### Cloud & Data Platform

* AWS
* Amazon S3
* Databricks

## Data

The initial project datasets include:

* **Rilevazione qualità aria 2024** — air-quality observations
* **Stazioni monitoraggio inquinanti** — monitoring-station metadata and coordinates
* **Historical weather data** — hourly meteorological observations
* **Uso e Copertura del Suolo 2021 (DUSAF 7.0)** — land-use and land-cover data

Raw and processed datasets are intentionally excluded from version control.

## Machine Learning Objective

The primary objective is to forecast future **PM2.5 concentrations** at monitoring stations using historical pollutant observations together with meteorological and spatial features.

The initial forecasting setup will use:

* **24-hour lookback window**
* **6-hour forecasting horizon**

Models:

1. RNN baseline
2. LSTM
3. GRU

Evaluation will include overall and spatiotemporal performance metrics such as:

* MAE
* RMSE
* R²
* MAE by monitoring station
* MAE by hour/day

## Project Structure

```text
Urban-Air-Quality-Platform/
│
├── airflow/
│   └── dags/
├── config/
├── data/
│   ├── raw/
│   ├── bronze/
│   ├── silver/
│   └── gold/
├── ingestion/
├── streaming/
├── preprocessing/
│   ├── bronze/
│   └── silver/
├── spark/
├── ml/
│   ├── datasets/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── inference/
├── db/
├── datahub/
├── notebooks/
├── tests/
├── docs/
├── docker/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── requirements-dev.txt
```

## Project Status

🚧 **In development**

Current phase:

* [x] GitHub repository initialized
* [x] Project structure created
* [x] Initial datasets identified
* [ ] Python environment and dependencies
* [ ] Data ingestion
* [ ] Bronze processing
* [ ] Silver processing
* [ ] Spatial feature engineering
* [ ] Gold feature pipeline
* [ ] Kafka streaming
* [ ] Airflow orchestration
* [ ] RNN/LSTM/GRU training
* [ ] MLflow experiment tracking
* [ ] DataHub lineage
* [ ] Databricks integration
* [ ] AWS deployment
