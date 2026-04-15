# 🫁 PneumoOps: Continuous MLOps Pipeline for 14-Class Thoracic Diagnostics

---

### 🎯 Project Task
An advanced, continuous ML/DL Ops pipeline designed for multi-label chest X-ray classification. The system containerizes models via **Docker** and deploys them to **Hugging Face Spaces**. It features live **A/B testing** between standard and optimized (ONNX) deployments, paired with robust **data drift monitoring** for 14 thoracic diseases, bridging the gap between lab research and continuous production.

---

### 🛠️ Focus on Deployment & Maintenance
Traditional clinical models natively degrade over time as scanner hardware and patient demographics shift. PneumoOps addresses this "Lab-to-Production" friction through proactive maintenance:
* **Real-time Latency Tracking:** Enables active comparison of model serving speeds, ensuring responsiveness on varying clinical hardware profiles in production environments.
* **Data Drift Detection:** Continuously compares incoming clinical image distributions against the original training baseline. Discovering out-of-distribution uploads instantly triggers a `Drift Detected` alert—providing the automated, programmatic trigger needed for model retraining cycles in true enterprise MLOps.

---

### 💻 Libraries / APIs
The pipeline is powered by a modern, full-stack open source MLOps ecosystem:
* **PyTorch** (Model Training & Baseline Backend)
* **ONNX Runtime** (Model Optimization & Accelerated Inference)
* **FastAPI** (High-performance API Routing & Drift Computation)
* **Gradio** (Interactive Maintenance Console & Dashboard)
* **Docker** (Environment Standardization & Containerization)
* **Hugging Face Hub/Spaces** (Cloud Hosting & Artifact Registry)

---

### 🏗️ Deployment Architecture
A streamlined, automated flow from data to deployment:
1. **Train** → Train Baseline MobileNetV3-small & Export to ONNX.
2. **Push to HF Hub** → Push version-controlled weights, training metrics, and code.
3. **Dockerized FastAPI Router** → Routes live traffic dynamically (50/50) for rapid A/B testing and computes drift statistics on the fly.
4. **Gradio UI Dashboard** → Empowers clinical data teams with an interactive maintenance dashboard showing top-3 predictions, model-arm comparisons, and system data-drift alerts.

---

### 📊 Datasets and Models
* **Dataset:** **ChestMNIST** — A highly efficient, 14-class multi-label dataset. Supports rapid CI/CD iteration loops while capturing complex thoracic conditions (e.g., Cardiomegaly, Pneumonia, Consolidation, Effusion, Mass).
* **Model A (Baseline):** **MobileNetV3-small** in standard PyTorch format, tailored for 14 output nodes.
* **Model B (Optimized):** **MobileNetV3-small** converted explicitly to ONNX architecture for faster execution speeds and reduced client hardware strain.
