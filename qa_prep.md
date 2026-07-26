# Q&A Preparation: Limitations & Future Work Roadmap

This document prepares concise response pairs (**Honest Answer** + **Next Steps Improvement**) for the 4 core technical limitations of the SentinelAI architecture, allowing you to proactively handle auditor questions during hackathon judging.

---

### Q1: "Did you tune your ensemble hyperparameters directly on the test set?"

- **Honest Answer:**  
  > *"Yes. Due to hackathon time constraints, ensemble parameters—such as the 0.99 percentile boost and the `res_recent >= 5` threshold—were tuned via iterative evaluation against the test split to maximize lateral movement recall without diluting static rules."*

- **What We'd Do Next:**  
  > *"In a production deployment, we would implement a formal 3-way data partition (60% Train / 20% Validation / 20% Held-Out Test) and use automated hyperparameter tuning (such as Optuna) exclusively on the validation set."*

---

### Q2: "How well does this synthetic dataset generalize to real-world enterprise SIEM telemetry?"

- **Honest Answer:**  
  > *"Our synthetic engine models realistic enterprise distributions (Beta login hours, Poisson session counts, Markov command sequences), but real corporate networks contain unpredictable noise—such as cloud microservice auto-scaling and ephemeral IP shifts."*

- **What We'd Do Next:**  
  > *"We would benchmark and fine-tune SentinelAI's feature extractors on open real-world SOC telemetry datasets, such as the Los Alamos National Lab (LANL) enterprise cybersecurity dataset and CERT insider threat logs."*

---

### Q3: "Why did the LSTM sequence autoencoder underperform on Insider Drift and Exfiltration?"

- **Honest Answer:**  
  > *"Sliding window sequence autoencoders ($W=10$) evaluate short-term command transitions, whereas insider drift and data exfiltration occur gradually over days or weeks, making them look statistically similar to normal reporting jobs in short windows."*

- **What We'd Do Next:**  
  > *"We would extend feature extraction with cumulative volume trajectory features (such as 7-day rolling data transfer ratios) and introduce Graph Neural Networks (GNNs) to track cross-entity data movement."*

---

### Q4: "Is Credential Stuffing detection recall high simply because of how the synthetic data was generated?"

- **Honest Answer:**  
  > *"Partly yes. Credential stuffing creates high-frequency failed authentication spikes that trigger deterministic Z-score rules cleanly, making static baseline recall exceptionally high (98.86%) by construction."*

- **What We'd Do Next:**  
  > *"We would test model robustness against low-and-slow adaptive adversaries who artificially throttle failed login attempts below Z-score thresholds by implementing adversarial attack evaluation loops."*
