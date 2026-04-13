# Multimodal-Graph-Learning-COVID19

This repository implements the main preprocessing, graph construction, modeling, and evaluation steps required to reproduce the results of the study:

**Arifi, D., Jain, D., Santillan, M., Resch, B. (2026)**  
*Comparative Evaluation of Geo-Social Media Signals in
Multimodal Graph-Based Epidemiological Forecasting*

---

## Overview

This project investigates how **multimodal geo-social media signals** can be integrated into **Graph Neural Networks (GNNs)** to improve short-term COVID-19 case forecasts at the US county level.

The study evaluates whether incorporating  
- geo-social media post time series,  
- user mobility-derived relational structures, and  
- semantic similarity between county-level post embeddings  

enhances predictive performance beyond traditional univariate epidemiological baselines and random control inputs.

Using an **A3T-GCN-based spatio-temporal forecasting framework** from Bai et al. (https://doi.org/10.3390/ijgi10070485), we compare multiple graph constructions and feature configurations to assess how different relational assumptions influence forecasting accuracy and spatial error patterns.

---

## Repository Contents

The repository includes:

- Preprocessed US county-level COVID-19 case time series  
- County-level geo-social media post aggregates  
- Model training and evaluation datasets and code for univariate and multivariate GNN variants  
- Scripts for spatial analysis of RMSE differences and population effects  
- Code to reproduce the main tables and figures in the paper  

The focus is on **short-term forecasting (up to 3 days ahead)**, **comparative model evaluation**, and **understanding how relational structure affects predictive performance across counties**.

---

## Methodological Summary

- COVID-19 case time series are aggregated at the US county level  
- Geo-social media posts are spatially assigned to counties and aggregated by day  
- Semantic similarity graphs are constructed using county-level post embeddings  
- User mobility graphs are derived from cross-county user movement inferred from geo-social media activity  
- Spatial distance graphs are constructed using county centroids  
- Random static and dynamic graphs serve as structural baselines  
- Models are trained using an A3T-GCN architecture with a 14-day input window and 3 day prediciton horizon
- Forecast performance is evaluated using RMSE and spatial error analysis  
- Population effects are assessed via regression between county population and RMSE differences  

All model comparisons are designed to isolate the effect of **input modality** and **graph structure**, rather than to optimize absolute forecasting accuracy.

---

## Reproducibility

All scripts are designed to reproduce the published results using the data provided in this repository.

Due to platform, privacy, and licensing constraints, raw geo-social media data cannot be redistributed. Instead, the repository includes **aggregated, anonymized, and derived datasets** sufficient to replicate all analyses.

Random seeds are fixed where applicable to ensure reproducibility of model training and evaluation.

---

## Citation

If you use this repository or build upon this work, please cite:

Arifi, D., Jain, D., Santillan, M., Resch, B. (2026).  
*Comparative Evaluation of Geo-Social Media Signals in
Multimodal Graph-Based Epidemiological Forecasting*

(Preprint or journal reference to be added upon publication.)

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or feedback, please contact:

**Dorian Arifi**  
Email: arifidorian.research@gmail.com
