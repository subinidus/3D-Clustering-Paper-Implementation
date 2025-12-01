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
📂 Project Structure
Plaintext

├── src/
│   ├── feature_extractor.py  # Extracts feature vectors from images using YOLO
│   └── visualizer.py         # Visualizes the extracted features using t-SNE
├── data/                     # Dataset directory (e.g., person, not_person)
├── results/                  # Output directory for CSVs and plots
└── README.md
🚀 Usage
1. Feature Extraction
Extract backbone features from your dataset.

Bash

python feature_extractor.py --data ./data/my_dataset --model yolov8s.pt --output ./results/features.csv
--data: Path to the dataset directory (must contain subfolders for each class).

--model: Name of the YOLO model (e.g., yolov8s.pt, yolov10s.pt).

--output: Path to save the extracted features (CSV).

2. Visualization (t-SNE)
Visualize the saved feature vectors.

Bash

python visualizer.py --features ./results/features.csv --output ./results/tsne_plot.png
--features: Path to the CSV file generated in step 1.

--output: Path to save the final t-SNE plot image.

📊 Results
Below is an example of the t-SNE visualization comparing 'Person' and 'Not-Person' classes.

(Place your result image here, e.g., ![t-SNE Result](./results/tsne_plot.png))

👨‍💻 Author
Seo Su-bin

Dept. of Computer Science (AI Computing), Kyungpook National University

Research Interest: Sports Data Science, Computer Vision, MLOps

📜 License
This project is licensed under the MIT License.
