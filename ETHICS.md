# ⚠️ ETHICS.md — PneumoOps Responsible AI Policy

> **This document is a mandatory part of the PneumoOps project and must be read before using or contributing to this system.**

---

## 🚨 1. NOT A Medical Device

> ❌ This tool does **NOT** detect disease with medical certainty.
> ✅ This tool **assists in prediction** as an educational MLOps demonstration.

PneumoOps is built for **academic and research purposes only**. It is a demonstration of MLOps concepts — A/B testing, drift monitoring, and continuous deployment — using chest X-ray classification as an applied example.

**This system is NOT:**
- A licensed medical device
- A substitute for professional clinical diagnosis
- Validated for use in patient care

**Every prediction must be treated as:**
- Potentially incorrect
- Requiring review by a qualified medical professional
- An educational output, not a clinical recommendation

---

## 🔐 2. Data Privacy & Consent

### What This System Does With Your Images
- Images uploaded to this tool are processed **in-memory** for inference only
- **No images are permanently stored** on the server by default
- **No user-identifiable information** (name, date of birth, patient ID, scan metadata) is collected, logged, or retained
- Prometheus metrics log **aggregate statistics only** (prediction class counts, latency) — never individual images or identities

### What You Must NOT Upload
- X-rays containing embedded patient metadata (DICOM tags)
- Images with visible patient names, dates, or hospital identifiers
- Any image you do not have the right to use

### If Data Collection Is Enabled (Optional Fine-Tuning Mode)
If the `PNEUMOOPS_COLLECT_DATA=true` environment variable is set:
- A text-based **consent notice** is shown to the user before submission
- Only the **anonymized image** (stripped of metadata) and its **prediction JSON** are saved
- Data is stored locally and never transmitted to third parties
- Users can opt out by not submitting their image

---

## ⚖️ 3. Bias & Accuracy Limitations

This model was trained on **ChestMNIST**, a research dataset with known limitations:

| Limitation | Risk |
|---|---|
| Small image size (224×224, grayscale) | May miss subtle findings visible on full-resolution clinical scans |
| Dataset demographic bias | Performance may vary across different patient populations, scanner hardware, and imaging protocols |
| Class imbalance | Rare conditions (Hernia, Pneumonia) have very low F1 scores (0.0) due to insufficient training examples |
| No temporal data | The model sees single frames only — cannot account for disease progression |
| No radiologist validation | Predictions have NOT been validated by clinical experts |

**Macro AUROC of 0.808** is a research-grade metric. It does **not** translate to clinical accuracy, sensitivity, or specificity at a diagnostic threshold.

---

## 🧠 4. Responsible AI Language Policy

All communication from this system must follow these rules:

### ✅ Use This Language
- "The model predicts a possible finding of..."
- "This assists in identifying potential..."
- "Confidence score indicates a statistical likelihood of..."
- "Results should be reviewed by a qualified clinician."

### ❌ Never Use This Language
- "This patient has pneumonia."
- "The model detects disease with X% accuracy."
- "This result confirms a diagnosis of..."
- "No disease found." (absence of prediction ≠ absence of disease)

---

## 🔄 5. Model Drift & Retraining Obligations

The drift monitoring system exists for a reason. When `DRIFT_DETECTED` is flagged:
- The incoming image is **statistically different** from training data
- Predictions on out-of-distribution data are **unreliable**
- A human reviewer should flag this image
- Retraining should be considered if drift is systematic

Ignoring persistent drift alerts in a production clinical system would be an **ethical failure**.

---

## 👥 6. Attribution & Accountability

| Role | Responsibility |
|---|---|
| Developers | Ensure model limitations are clearly communicated |
| Operators | Never deploy without visible disclaimers |
| Users | Never use predictions to make unsupervised clinical decisions |
| Evaluators | Judge on MLOps pipeline quality, not clinical validity |

---

## 📋 7. Compliance Acknowledgment

By using, deploying, or contributing to PneumoOps, you acknowledge that:

- [ ] You have read this document in full
- [ ] You understand this is an educational tool, not a medical device
- [ ] You will not use predictions as a substitute for clinical judgment
- [ ] You will not upload images containing patient PII
- [ ] You accept responsibility for any use of this system

---

*This ethics policy was authored as part of the PneumoOps academic MLOps project. It follows principles from the [EU AI Act](https://www.europarl.europa.eu/topics/en/article/20230601STO93804/eu-ai-act-first-regulation-on-artificial-intelligence), [WHO Ethics Guidelines for AI in Health](https://www.who.int/publications/i/item/9789240029200), and [Google's Responsible AI Practices](https://ai.google/responsibility/responsible-ai-practices/).*
