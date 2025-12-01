# Visual Comparative Analysis of YOLO Backbone Feature Extraction
## 📌 Introduction
This repository contains the official implementation code for the research project: **"Comparative Analysis of Backbone Feature Extraction Performance in YOLOv8 and YOLOv10 Models."**

This project focuses on analyzing the feature embedding capabilities of different YOLO architectures. Instead of using the final detection output, we extract high-dimensional feature vectors directly from the backbone and visualize their clustering performance using t-SNE. This approach provides insights into how well the models distinguish between classes (e.g., Person vs. Not-Person) in complex environments such as sports fields.

## ✨ Key Features
* **Backbone Feature Extraction**: Extracts 1D feature vectors from YOLO models (v8, v10, etc.) using `ultralytics` API.
* **Dimensionality Reduction**: Implements t-SNE (t-Distributed Stochastic Neighbor Embedding) to project high-dimensional features into 2D space.
* **Class Separability Visualization**: Visualizes the clustering performance to qualitatively evaluate the backbone's feature representation.
* **Modular Design**: Separated feature extraction and visualization modules for flexibility.

## 🛠️ Requirements
* Python 3.8+
* ultralytics
* pandas
* scikit-learn
* matplotlib
* tqdm

To install requirements:
```bash
pip install ultralytics pandas scikit-learn matplotlib tqdm
