**Machine Learning-Based Network Intrusion Detection System**

**Project Overview**
This project focuses on detecting malicious network traffic using machine learning techniques on the CICIDS2017 dataset. The goal is to classify network traffic as **benign or attack** and improve detection performance using multiple models.

**Objectives**
- Detect intrusion in network traffic
- Compare baseline and advanced ML models
- Analyze performance using metrics
- Perform ablation study and error analysis

**Dataset**
- Dataset: CICIDS2017
- Contains real-world network traffic data
- Includes both benign and attack records

**Models Used**
- **Logistic Regression** (Baseline)
- **Random Forest** (Advanced Model)

**Evaluation Metrics**
- Accuracy  
- Precision  
- Recall  
- F1 Score  

**Key Features**
- Data preprocessing and cleaning  
- Exploratory Data Analysis (EDA)  
- Model training and evaluation  
- Model comparison  
- Ablation study (raw vs scaled data)  
- Error analysis (misclassification study)  

**Results**
- High accuracy achieved (~99%)  
- Random Forest performs slightly better in handling complex patterns  
- Minimal impact of scaling observed  

**Team Members**
- Nityavardhan Singh – Team Coordinator  
- Kartik Arora – Data Engineer  
- Keshav Goyal – ML Engineer  
- Shreya Mittal – Documentation & Presentation

**How to Run**

1. Install dependencies:

bash
pip install -r requirements.txt

2. Run Week 2 pipeline:
python run_week2.py

3. Run Week 3 pipeline:
python run_week3.py
