# SentinelAI: Tiered Monotonic Hybrid Ensemble Architecture for Enterprise Anomaly Defense

[![Live Dashboard](https://img.shields.io/badge/Live_Dashboard-Operational-00f0ff?style=for-the-badge&logo=vercel)](https://honeywell-seven.vercel.app/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![PR-AUC](https://img.shields.io/badge/Core_PR--AUC-0.7475-00ff66?style=for-the-badge)](FINAL_REPORT.md)
[![Hackathon](https://img.shields.io/badge/SIH_2026-Task_4-ffaa00?style=for-the-badge)]()

> **Honeywell Problem Statement: Task 4**  
> **Problem Statement Title:** Behavioral Anomaly Detection & Threat Hunting in Multi-Entity Enterprise Telemetry  
> **Theme:** Cyber Security & AI Operations | **Category:** Software  
> **Team Lead:** Raj Verma | **Registration ID:** 23BAI10806  

---

## 🌐 Live Interactive SOC Command Center

Access the deployed glassmorphic Cyber SOC Dashboard online:  
👉 **[https://honeywell-seven.vercel.app/](https://honeywell-seven.vercel.app/)**

---

## 📌 Executive Summary & Project Overview

Modern enterprise Security Operations Centers (SOCs) face a critical dilemma:
1. **Static Policy Rules** are highly precise for perimeter breaches (such as credential stuffing or impossible travel), but miss multi-step internal network traversal (lateral movement).
2. **Standard Unsupervised Machine Learning** models (such as Isolation Forest or deep sequence autoencoders) output continuous reconstruction loss. When naively combined, background noise leapfrogs deterministic policy breaches, crowding out high-confidence alerts in tight SOC triage queues.

**SentinelAI** solves this challenge by engineering a **Tier-Gated Monotonic Hybrid Ensemble Architecture**. By establishing a strict rule floor ($\text{BaseScore} \ge 3.0$) and applying a sequence boost only to entities exhibiting multi-step resource traversal ($\text{res\_recent} \ge 5$), SentinelAI guarantees **zero performance drop on static rules** while deploying PyTorch LSTM sequence autoencoders to capture stealthy lateral movement attacks.

---

## 📊 Authoritative Performance Benchmarks

Evaluated over 60 days of enterprise telemetry ($N = 145,290$ total events; $N = 28,758$ ground-truth test split; $N = 653$ core anomalies):

| Architecture / Model | Core PR-AUC | Top 1.0% Budget Recall ($N=287$) | Top 2.5% Budget Recall ($N=718$) | Lateral Movement Recall (Top 2.5%) |
|---|---|---|---|---|
| **Static Baseline Profiler** | 0.7201 | 72.43% (473/653) | 79.79% (521/653) | 44.16% (34/77) |
| **Isolation Forest (Tabular ML)** | 0.2980 | 22.82% (149/653) | 51.61% (337/653) | 3.90% (3/77) |
| **LSTM Autoencoder (Sequence ML)** | 0.1406 | 6.13% (40/653) | 11.18% (73/653) | 24.68% (19/77) |
| **SentinelAI Monotonic Hybrid Ensemble** | **0.7475** | **72.74% (475/653)** | **81.32% (531/653)** | **51.95% (40/77)** |

### Key Achievements:
- **+6 Additional Lateral Movement Catches:** Lifts lateral movement recall from **44.16% (34/77) to 51.95% (40/77)** at the Top 2.5% budget (~12 alerts/day).
- **81.32% Core Anomaly Recall:** Flags 4 out of 5 real cyber attacks within a strict 2.5% analyst alert budget.
- **Zero Static Rule Degradation:** 100% precision match on static rule categories (Credential Stuffing `174/176` = 98.86%, Impossible Travel `75/75` = 100.00%, Brute Force `137/141` = 97.16%).

---

## 🧮 Mathematical Hybrid Formulation

$$\text{HybridScore} = \begin{cases} \text{BaselineScore} & \text{if } \text{BaselineScore} \ge 3.0 \\ \text{BaselineScore} + 0.99 \times \text{LSTM}_{\text{pct}} & \text{if } \text{res\_recent} \ge 5 \\ \text{BaselineScore} & \text{otherwise} \end{cases}$$

Where:
- $\text{BaselineScore}$: Scoring output from the multi-attribute statistical profiler ($0.0 - 5.5$).
- $\text{LSTM}_{\text{pct}}$: Percentile rank of the PyTorch LSTM sequence autoencoder reconstruction error ($[0, 1]$).
- $\text{res\_recent} \ge 5$: Traversal threshold checking distinct resource access count in sliding windows ($W=10$).

---

## 🏗 System Architecture & Pipeline

```
  ┌──────────────────────────────────────────────────────────┐
  │                 Multi-Entity Telemetry Ingestion         │
  │     (145,290 events / 60 Days / Users & Service Accts)  │
  └─────────────────────────────┬────────────────────────────┘
                                │
          ┌─────────────────────┴─────────────────────┐
          ▼                                           ▼
┌───────────────────────────┐               ┌───────────────────────────┐
│     Baseline Profiler     │               │   LSTM Sequence Engine    │
│  Z-scores & Heuristics    │               │  Window W=10 Autoencoder  │
│ (Hour, Geo, Device, Res)  │               │   Reconstruction Error    │
└─────────┬─────────────────┘               └───────────┬───────────────┘
          │                                             │
          │             ┌───────────────────────────────┘
          ▼             ▼
┌───────────────────────────────────────────────────────────┐
│           Tier-Gated Monotonic Hybrid Scorer              │
│    Preserves Base Rule Floor (>=3.0) + Traversal Boost    │
└───────────────────────────┬───────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│            Glassmorphic Cyber SOC Command Center          │
│       Live Analytics, Incident Narratives & AI Triage     │
└───────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
Honeywell/
├── index.html                   # Main static web application entrypoint (Root)
├── styles.css                   # Glassmorphic neon CSS design system (Root)
├── app.js                       # Dashboard logic, Chart.js & replay engine (Root)
├── package.json                 # Static Vercel deployment configuration
├── vercel.json                  # Clean static URL routing settings
├── .vercelignore                # Deployment exclusions
├── FINAL_REPORT.md              # Authoritative technical whitepaper & evaluation
├── pitch_talking_points.md      # Executive business-language translation
├── naive_ensemble_story.md      # 90-second spoken script on ensemble journey
├── qa_prep.md                   # Hackathon Q&A defense card
├── demo_case_study.md           # Before/After case study (user_018)
├── generate_data.py             # Phase 1: Synthetic telemetry generation engine
├── run_baseline.py              # Phase 2: Static baseline statistical profiler
├── run_phase3.py                # Phase 3/4: ML training, ensemble scoring & feed export
├── dashboard/                   # Secondary dashboard static asset folder
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── data/                        # Processed telemetry and model feeds
│   ├── dashboard_feed.json      # Top 1,000 alert JSON feed for frontend
│   ├── events.csv
│   └── ground_truth.csv
├── datagen/                     # Telemetry generation modules & attack injectors
├── ml/                          # PyTorch LSTM, Isolation Forest & Ensemble Scorer
└── docs/                        # Hackathon presentation artifacts
    ├── SentinelAI_SIH_Presentation.pptx   # Official SIH 6-slide presentation deck
    ├── SentinelAI_SIH_Presentation.pdf    # Official SIH PDF presentation export
    └── IDEA_Presentation_Format.pptx      # Template format PPTX
```

---

## 💻 Interactive Dashboard Features

- **Glassmorphic Cyber Aesthetics:** Dark-mode UI with cyan/magenta glows, smooth micro-animations, and Google Fonts (`Outfit`, `Inter`, `JetBrains Mono`).
- **Live Indian Standard Time (IST) Clock:** Live header clock formatted to `Asia/Kolkata` time.
- **Interactive Chart.js Suite:** Model comparison bar charts, per-attack recall spectrum, and 60-day temporal threat timeline.
- **Automated AI Incident Narratives:** Generates human-readable SOC explanations for every escalation in the triage console.
- **Feature Attribution Modal:** Detailed breakdown of how Static Baseline Risk, LSTM Risk, and Tabular ML contributed to composite risk.
- **⚡ Replay Live Attack Engine:** Interactive streaming replay button that simulates live telemetry ingestion, flashing highlights when `user_018` lateral movement attack streams in.

---

## 🛠 Local Installation & Execution

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/irajverma/Honeywell.git
cd Honeywell
pip install -r requirements.txt
```

### 2. Run Complete ML Pipeline (Phase 1 to 4)
```bash
# Step 1: Generate synthetic 60-day telemetry dataset (145,290 events)
python generate_data.py

# Step 2: Execute static baseline statistical profiler
python run_baseline.py

# Step 3: Train LSTM autoencoder, Isolation Forest, run ensemble & export feed
python run_phase3.py
```

### 3. Launch Local Dashboard Web Server
```bash
# Start local HTTP server on port 8080
python -m http.server 8080
```
Open your browser and navigate to: **`http://localhost:8080/`**

---

## 📄 Hackathon Presentation & Documentation

- **Live Deployed Dashboard:** [https://honeywell-seven.vercel.app/](https://honeywell-seven.vercel.app/)
- **Technical Whitepaper:** [`FINAL_REPORT.md`](FINAL_REPORT.md)
- **Official SIH Presentation PPTX:** [`docs/SentinelAI_SIH_Presentation.pptx`](docs/SentinelAI_SIH_Presentation.pptx)
- **Official SIH Presentation PDF:** [`docs/SentinelAI_SIH_Presentation.pdf`](docs/SentinelAI_SIH_Presentation.pdf)
- **Pitch Talking Points:** [`pitch_talking_points.md`](pitch_talking_points.md)
- **Ensemble Story Script:** [`naive_ensemble_story.md`](naive_ensemble_story.md)
- **Q&A Defense Card:** [`qa_prep.md`](qa_prep.md)
- **Demo Case Study:** [`demo_case_study.md`](demo_case_study.md)

---

## 🤝 Team & Acknowledgments

- **Hackathon:** Smart India Hackathon (SIH) 2026
- **Problem Statement:** Task 4 — Behavioral Anomaly Detection & Threat Hunting in Multi-Entity Enterprise Telemetry
- **Team Lead:** Raj Verma (Registration ID: `23BAI10806`)
- **Organization:** Honeywell Cyber Security Operations Center (SOC) Engineering Project

*Built with Python 3.12, PyTorch, Vanilla JavaScript, Chart.js & Vercel.*
