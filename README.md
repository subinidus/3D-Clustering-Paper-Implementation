# Visual Comparative Analysis of YOLO Backbone Feature Extraction

## 📌 Introduction
This repository contains the official implementation code for the research project: **"Comparative Analysis of Backbone Feature Extraction Performance in YOLOv8 and YOLOv10 Models."**

This project focuses on analyzing the feature embedding capabilities of different YOLO architectures. Instead of using the final detection output, we extract high-dimensional feature vectors directly from the backbone and visualize their clustering performance using t-SNE. This approach provides insights into how well the models distinguish between classes (e.g., Person vs. Not-Person) in complex environments such as sports fields.

## ✨ Key Features
* **Backbone Feature Extraction**: Extracts 1D feature vectors directly from YOLO models using the `ultralytics` API.
* **Dimensionality Reduction**: Implements t-SNE to project high-dimensional features into 2D space.
* **Qualitative Analysis**: Visualizes class separability to evaluate feature representation power.
* **Modular Design**: Separates extraction and visualization logic for scalability.

## 🛠️ Requirements
* Python 3.8+
* ultralytics
* pandas
* scikit-learn
* matplotlib
* tqdm

To install requirements:
```
pip install ultralytics pandas scikit-learn matplotlib tqdm
```

## 📂 Project Structure

├── src/
│   ├── feature_extractor.py  # Script to extract features from images
│   └── visualizer.py         # Script to visualize features using t-SNE
└── README.md

![Pipeline](./assets/paper_pipeline.png)

## 🚀 Usage
1. Feature Extraction
Extract backbone features from your dataset and save them as a CSV file.

```
# Run from the project root directory
python src/feature_extractor.py --data ./data --model yolov8s.pt --output ./results/features.csv
```
--data: Path to the dataset directory (must contain subfolders like person, not_person).

--model: Name of the YOLO model (e.g., yolov8s.pt, yolov10s.pt).

--output: Path to save the extracted features CSV.

2. Visualization (t-SNE)
Visualize the extracted feature vectors.

```
python src/visualizer.py --features ./results/features.csv --output ./results/tsne_plot.png
```
--features: Path to the CSV file generated in step 1.

--output: Path to save the final t-SNE plot image.

## 📊 Results
Below is an example of the t-SNE visualization comparing 'Person' and 'Not-Person' classes.

(Run the code to generate your own tsne_plot.png and it will appear here)

## 👨‍💻 Author
Seo Su-bin

Dept. of Computer Science (AI Computing), Kyungpook National University

Research Interest: Sports Data Science, Computer Vision

## 📜 License
This project is licensed under the MIT License.

Research Interest: Sports Data Science, Computer Vision


## 📜 License
This project is licensed under the MIT License.
