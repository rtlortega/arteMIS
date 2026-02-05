# arteMIS

> **Work in progress (pre-release) 🚧**  
> arteMIS is a statistically gorunded framework for **robust, reproducible, and interpretable** molecular networking. It is desgined to help users to **tune parameters, score network quality, and compare runs** by measuring how well network topology agrees with chemistry based metrics. 

[![Status](https://img.shields.io/badge/status-WIP-orange.svg)](#) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](#license) [![Python](https://img.shields.io/badge/python-3.10%2B-informational.svg)](#requirements)

---

## ✨ Key ideas

- **Parameter tuning & benchmarking:** Compare MN runs (e.g., spectral similarity metrics, diverse thresholds, min peaks, max links) with validated evaluation metrics.  
- **Statistcal optimization:** Define optimal configuration for your MN experiments (e.g., targted mining of terpenoids).  
- **Topology ↔ Chemistry agreement:** Quantifies how network structure (edges/nodes/components) aligns with chemistry-derived groupings or distances (e.g., morgan fingerprints and tanimoto similarity).


> arteMIS does run spectral networking itself. You can **evaluate** the optimized configurations in your own dataset using matchMS-based metrics.

---

## 📦 Requirements
Nose hehe

---

## 🧩 Metrics (current & planned) 


**Implemented (early stage):**
- Component/family **purity** vs. chemical classes  
- **Neighborhood consistency** vs. chemistry similarity  
- **Ranked run score** (composite of selected metrics)
- Fingerprint-aware **entanglement** of topology vs. chemistry dendrograms

**Planned:**

---

## 🤝 Contributing
Since this is work in progress, contributions are very welcome! Please open an issue to discuss features/bugs.
---


