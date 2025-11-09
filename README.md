# 🧠 Agentic Doctor: Modular AI-Powered Medical Assistant (CrewAI)

Agentic Doctor is a **multimodal, agentic AI system** built with [CrewAI](https://docs.crewai.com/) and [Streamlit](https://streamlit.io/) that performs **medical image analysis**, **lab report interpretation**, **PubMed research retrieval**, and **automated PDF report generation**.  

This project demonstrates how **multi-agent collaboration** can simulate a team of digital doctors — radiologists, lab analysts, and research assistants — producing structured, explainable, and ethical diagnostic summaries.

---

## 🩺 Overview

| Feature | Description |
|----------|-------------|
| **Framework** | CrewAI (Multi-Agent Orchestration) |
| **Frontend** | Streamlit |
| **Capabilities** | Image parsing, OCR, PubMed research, PDF report |
| **Outputs** | Diagnosis summary, citations, and patient-safe recommendations |
| **Ethics** | Educational only — *not for medical use* |

---

## 📊 System Architecture (Graph View)

```text
                          ┌────────────────────────────┐
                          │       Streamlit UI         │
                          │   (File Upload + Results)  │
                          └────────────┬───────────────┘
                                       │
                                       ▼
                          ┌────────────────────────────┐
                          │  CrewAI Orchestrator       │
                          │  (src/main.py)             │
                          └────────────┬───────────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          ▼                            ▼                            ▼
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  Radiology Agent     │     │   Lab Report Agent   │     │   Research Agent     │
│  (Image Analysis)    │     │  (OCR + Lab Values)  │     │ (PubMed Retrieval)   │
│  → DICOM, PNG, JPG   │     │  → PDF, TXT, JPG     │     │  → Relevant Studies  │
└──────────┬───────────┘     └──────────┬───────────┘     └──────────┬───────────┘
           │                             │                             │
           └───────────────┬─────────────┴───────────────┬─────────────┘
                           ▼                             ▼
                 ┌───────────────────────────────┐
                 │        Report Agent           │
                 │  (PDF Generation + Summary)   │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │      Output Reports           │
                 │  → PDF, Diagnosis, Citations  │
                 └───────────────────────────────┘
```

# ⚙️ Agentic Doctor — Workflow and Configuration Details

---

## ⚙️ Workflow Summary

| Step | Component | Description |
|------|------------|-------------|
| 1️⃣ | **Frontend (Streamlit)** | Upload lab or scan file. |
| 2️⃣ | **Radiology Agent** | Analyze X-ray/CT images and detect abnormalities. |
| 3️⃣ | **Lab Agent** | Extract lab results and interpret anomalies. |
| 4️⃣ | **Research Agent** | Retrieve PubMed studies supporting the diagnosis. |
| 5️⃣ | **Report Agent** | Generate structured, explainable PDF summary. |

---
# 📤 Agentic Doctor — Upload, Output, and System Overview

---

## 📤 Upload Options

Upload the following file types to start your AI-powered diagnosis workflow.

### 🩻 Medical Scans
- `.dcm` — DICOM (X-ray, MRI, CT)
- `.nii` — NIfTI medical format
- `.png`, `.jpg` — Standard medical images

### 🧬 Lab Reports
- `.pdf` — Digital lab reports
- `.txt` — Text-based reports
- `.png`, `.jpg` — Scanned or image-based reports

---

## 📊 Output Includes

Your uploaded files are processed by **CrewAI agents** to produce the following results:

- ✅ **Diagnosis Summary** — AI interpretation of the condition  
- 🧠 **Lab Findings** — Key abnormalities and numerical deviations  
- 🔬 **PubMed Citations** — Related research studies  
- 📄 **Downloadable PDF Report** — Consolidated and explainable report  

---

## 🧪 Example Output

```text
Diagnosis Summary:
   Condition: Pneumonia (High Confidence)
   Observations: Bilateral infiltrates on chest X-ray
   Lab Results: Elevated WBC, neutrophilia
   PubMed References: 3 supporting studies (2024–2025)
   Recommendation: Consult physician, start antibiotics

PDF Report:
   → /reports/patient_case_23.pdf

```

# 🧰 Tech Stack

| Layer | Technology |
|-------|-------------|
| **Frontend** | Streamlit |
| **AI Framework** | CrewAI |
| **Models** | GPT-4o-mini (OpenAI), Claude (configurable) |
| **Vision** | OpenCV, pydicom, SimpleITK |
| **OCR** | pytesseract |
| **Research** | BioPython, Entrez API |
| **Reports** | ReportLab / FPDF |
| **Python** | 3.11 |

---

## ⚖️ Ethical Disclaimer

> ⚠️ **Note:**  
> This system is developed **strictly for educational and research purposes**.  
> It is **not intended to replace licensed medical professionals** or certified diagnostic tools.  
> Always consult a qualified doctor before making any medical decisions.

---

## 👨‍💻 Author

**Sohan Kumar Shah**  
🎓 Final Year B.Tech (CSE) — KIIT University  
📧 **mail.sohankrshah@gmail.com**

---

🧠 *Empowering Healthcare through Responsible AI.*
