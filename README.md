# MixxComm — AI-Powered Customer Churn Ecosystem 🚀

**Turning 594,000+ data records into actionable business strategies.**

MixxComm adalah platform analitik dan prediksi *customer churn* yang dirancang untuk membantu tim manajemen memahami mengapa pelanggan pergi dan bagaimana cara menahannya. Platform ini mengintegrasikan *Machine Learning*, *Microservices Architecture*, dan *Generative AI* dalam satu ekosistem yang kohesif.

---

## 🌟 Key Features

### 1. 📊 Interactive Analytics Dashboard
Dashboard modern yang memvisualisasikan metrik kunci seperti *Churn Rate*, *Tenure Cohorts*, dan *Top Churn Drivers*. Menggunakan **Plotly** untuk grafik interaktif yang membantu identifikasi "zona merah" pelanggan.

### 2. 🛠️ Churn Prevention Simulator (The Killer Feature)
Bukan sekadar prediksi statis. Fitur ini memungkinkan tim marketing mensimulasikan perubahan pada variabel kontrol (seperti *Monthly Charges* atau jenis *Contract*) untuk melihat dampak langsung terhadap probabilitas churn pelanggan secara *real-time*.

### 3. 🤖 Rini — AI Business Consultant
Asisten berbasis **RAG (Retrieval-Augmented Generation)** yang ditenagai oleh OpenAI. Rini mampu menjawab pertanyaan strategis, melakukan analisis data menggunakan *pandas* di latar belakang, dan memberikan rekomendasi retensi personal melalui antarmuka chat.

### 4. 🔗 Microservices Architecture
Sistem dipisahkan menjadi dua bagian utama:
- **Backend (FastAPI):** Bertugas mengurus semua logika *Machine Learning* dan *Preprocessing* melalui REST API.
- **Frontend (Streamlit):** Interface modular yang ringan dan fokus pada pengalaman pengguna.

---

## 🛠️ Tech Stack

- **Languages:** Python 3.10+
- **Machine Learning:** XGBoost, Scikit-learn, Imbalanced-learn (Random OverSampling).
- **Backend API:** FastAPI, Uvicorn, Pydantic.
- **Frontend UI:** Streamlit, Plotly.
- **AI/LLM:** OpenAI GPT-4o, ChromaDB (Vector Store), RAG Architecture.
- **Data:** Pandas, NumPy.

---

## 📂 Project Structure

```text
customer_churn/
├── main.py             # Entry point aplikasi (Frontend)
├── app.py              # Backend API (FastAPI)
├── run.py              # Orchestrator script untuk menjalankan sistem
├── views/              # Modular UI Components
│   ├── ui_analytics.py # Charts & Analytics
│   ├── ui_predict.py   # Individual API Prediction
│   ├── ui_agent.py     # AI Consultant Interface
│   └── simulator.py    # Business Simulation Logic
├── artifacts/          # ML Models & Preprocessors (.pkl)
├── data/               # Raw & Processed Datasets
├── llm/                # AI Agent & RAG Logic
└── requirements.txt    # Project Dependencies
```

---

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   https://github.com/Spyrx4/customer_churn.git
   cd customer-churn
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the entire system (Backend + Frontend):**
   ```bash
   python run.py
   ```
   *Backend akan berjalan di port 8000, dan Frontend akan terbuka otomatis di browser.*

---

## 🧠 Engineering Philosophy: Human-in-the-loop

Project ini dikembangkan dengan filosofi bahwa AI adalah **Co-pilot**. Meskipun AI digunakan sebagai akselerator koding dan generator logika awal, **setiap baris kode dan keputusan arsitektur telah melalui proses audit, validasi, dan pengawasan manusia secara ketat.** Ini memastikan integritas sistem dan keakuratan hasil prediksi di tingkat produksi.

---

## 🗺️ Roadmap
- [x] Refactor Microservices (FastAPI & Streamlit)
- [x] Implementation of Churn Prevention Simulator
- [ ] Integration of SHAP Values for Explainable AI (XAI)
- [ ] Real-time IP-to-Location Enrichment for Fraud Analytics
- [ ] Deployment to Production (Render/Streamlit Cloud)

---
**Erick — AI-Augmented Engineer**
*"Bridging the information gap with humanized technology."*
