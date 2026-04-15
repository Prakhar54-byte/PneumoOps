# PneumoOps Assignment Target

## Title
PneumoOps: Continuous MLOps Pipeline with A/B Testing & Drift Monitoring for Multi-Class Thoracic Diagnostics

## Project Task
Build an advanced, continuous ML/DL Ops pipeline for 14-class chest X-ray classification. The system will deploy containerized models via Docker on Hugging Face Spaces, featuring live A/B testing between a standard and an optimized (ONNX) model. With a strict focus on real-world maintenance, the backend includes real-time inference latency tracking and a data drift monitoring system that triggers automated alerts for model retraining when production data shifts.

## Idea Explanation in Detail
This project upgrades a multi-label classification task into a complete, production-ready MLOps system, addressing real-world "Lab-to-Production" challenges and ongoing maintenance. In a real clinical setting, models degrade over time as scanner hardware changes or patient demographics shift. Instead of deploying a single static model, we upload two versions to the Hugging Face Hub: a baseline PyTorch model and an ONNX-optimized model.
The Dockerized FastAPI backend performs A/B testing by randomly routing user uploads to either model, allowing us to compare real-world inference latency and performance. Crucially for deployment maintenance, the backend includes a Data Drift Monitor that compares the incoming image's statistical distribution against the training dataset. If an out-of-distribution image is uploaded, the system flags it as "Drift Detected," simulating the exact trigger required for automated retraining cycles in enterprise MLOps.

## Datasets and Models to be used

- **Dataset**: ChestMNIST (14-class multi-label dataset containing Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax). It is lightweight enough for rapid CI/CD iterations without wasting hours on training.
- **Model A (Baseline)**: MobileNetV3-small (Standard PyTorch format adapted for 14 output nodes).
- **Model B (Optimized)**: MobileNetV3-small (Converted to ONNX format for faster client-level inference).

## Task 1: The Gradio Web UI
The UI needs to have an image upload component for a chest X-ray. When the user clicks submit, the app should send the image via a `requests.post` call to a local FastAPI backend (assume the URL is `http://127.0.0.1:8000/predict`).

The UI needs to display the following output fields clearly:
- **Predictions**: A bar chart or clean list showing the top 3 highest-probability conditions out of the 14 thoracic disease classes, e.g., Cardiomegaly, Effusion, Mass
- **Model Used**: This will show either 'Baseline PyTorch' or 'Optimized ONNX' for our A/B testing
- **Inference Latency**: (ms)
- **Data Drift Alert**: Will show 'Normal' or 'Drift Detected' with a warning color to simulate production maintenance alerts

*Please make the UI look clean, professional, and add a nice title and description at the top explaining that this is an MLOps A/B Testing Pipeline focused on real-world deployment and reliability.*

## Task 2: The Poster Text
Draft the text content for an academic project poster. Title: PneumoOps: Continuous MLOps Pipeline for 14-Class Thoracic Diagnostics. Generate professional, concise text for the following sections:
- **Project Task**
- **Focus on Deployment & Maintenance**
- **Libraries / APIs**
- **Deployment Architecture**
- **Datasets and Models**
*Keep the text punchy and readable for a visual poster presentation.*
