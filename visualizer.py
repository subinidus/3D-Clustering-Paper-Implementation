import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

def visualize_features(feature_file, output_image):
    
    print(f"Reading feature file: '{feature_file}'...")
    try:
        df = pd.read_csv(feature_file)
    except FileNotFoundError:
        print(f"Error: File not found at '{feature_file}'")
        return

    if df.empty:
        print("Error: The feature file is empty.")
        return

    labels = df['label']
    features = df.drop(columns=['filename', 'label'])

    print("Running t-SNE... (This may take a moment)")

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(features)

    df_tsne = pd.DataFrame(tsne_results, columns=['x', 'y'])
    df_tsne['label'] = labels
    
    colors = {'person': 'red', 'not_person': 'blue'}
    
    plt.figure(figsize=(12, 10))
    
    for label, color in colors.items():
        subset = df_tsne[df_tsne['label'] == label]
        plt.scatter(subset['x'], subset['y'], c=color, label=label, alpha=0.6)
    
    plt.legend()
    plt.title(f't-SNE Visualization ({os.path.basename(feature_file)})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    

    plt.savefig(output_image)
    print(f"Visualization complete. Saved plot to '{output_image}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t-SNE Feature Visualizer")
    parser.add_argument('--features', type=str, required=True, help="Input CSV feature file")
    parser.add_argument('--output', type=str, required=True, help="Output image file name (e.g., tsne_plot.png)")
    
    args = parser.parse_args()
    
    visualize_features(args.features, args.output)