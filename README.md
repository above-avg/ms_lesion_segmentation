# Automated MS Lesion Segmentation via Multi-modal 2.5D UNet++

**Authors:** Pallab Sarangi, Shreyas Baloni  
**Affiliation:** RNS Institute of Technology (RNSIT)   
**License:** Apache 2.0

---

### Quantitative Performance Overview
![FLAIR](results/clinical_sample_50.png)

> **Note:** The above visual demonstrates the high specificity of the ensemble model in the periventricular regions, successfully distinguishing lesions from cerebrospinal fluid signals.

---

## Abstract
This repository contains the implementation of a State-of-the-Art (SOTA) deep learning framework for the automated segmentation of Multiple Sclerosis (MS) lesions. By evolving through eight architectural iterations—from a standard 2D U-Net to a Nested UNet++ with Deep Supervision—this system achieves a **Total Lesion Load (TLL) Correlation of $r = 0.9063$** ($p < 2.34 \times 10^{-58}$). 

The system utilizes a 2.5D multi-slice stacking approach combined with multi-modal sequence fusion (FLAIR, T1, T2) to provide the model with essential spatial and contrast context, significantly reducing false positives near the ventricles.

## Architectural Evolution
The project followed a rigorous research trajectory to address clinical bottlenecks:
1. **Baseline Phase:** Standard 2D U-Net (Dice: 0.3491).
2. **Attention Phase:** Integration of Spatial Attention Gates to suppress healthy tissue noise (Dice: 0.4790).
3. **SOTA Phase:** Nested UNet++ with 2.5D context and 5-Fold Ensemble (Dice: 0.5949, Correlation: 0.91).

## Key Research Results
* **Clinical Accuracy:** $r = 0.9063$ volumetric correlation with human gold standards.
* **Sensitivity Control:** Adjustable sensitivity up to 72.19% for early-stage screening.
* **Reliability:** 99.91% Specificity, ensuring negligible false-positive rates in clinical reports.

## Repository Structure
* `src/`: Core Python implementation for preprocessing, training, and ensemble evaluation.
* `checkpoints/`: Trained model weights for all 5 folds (Stored via Git LFS).
* `results/`: Quantitative reports and visual error analysis (False Negative Analysis).

## Installation and Usage
1. Clone the repository: `git clone https://github.com/above-avg/ms_lesion_segmentation.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Prepare data: `python src/sota_prep.py`
4. Run inference: `python src/sota_eval.py`

---
**Citation:** If you use this work in your research, please cite the RNSIT/SJRI MS Research Proposal (2026).
