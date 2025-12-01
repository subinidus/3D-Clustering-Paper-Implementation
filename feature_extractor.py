import os
import argparse
import pandas as pd
from ultralytics import YOLO
from glob import glob
from tqdm import tqdm

def extract_features(data_dir, model_name, output_file):
    
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return

    person_images = glob(os.path.join(data_dir, 'person', '*.jpg'))
    not_person_images = glob(os.path.join(data_dir, 'not_person', '*.jpg'))
    
    print(f"Found {len(person_images)} person images and {len(not_person_images)} not_person images.")

    all_images = []
    all_images.extend([(img, 'person') for img in person_images])
    all_images.extend([(img, 'not_person') for img in not_person_images])

    results = []

    print("Extracting features...")
    for img_path, label in tqdm(all_images):
        try:
            result = model.embed(source=img_path, save=False, verbose=False)
            
            feature_vector = result[0].cpu().numpy().flatten()
            
            results.append([os.path.basename(img_path), label] + list(feature_vector))
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if not results:
        print("No features were extracted. Check data directory or model.")
        return

    feature_size = len(results[0]) - 2
    columns = ['filename', 'label'] + [f'feat_{i}' for i in range(feature_size)]
    
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(output_file, index=False)
    
    print(f"Feature extraction complete. Saved {len(df)} results to '{output_file}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Backbone Feature Extractor")
    parser.add_argument('--data', type=str, required=True, help="Dataset directory (containing person/not_person subfolders)")
    parser.add_argument('--model', type=str, required=True, help="YOLO model name (e.g., yolov8s.pt)")
    parser.add_argument('--output', type=str, required=True, help="Output CSV file name")
    
    args = parser.parse_args()
    
    extract_features(args.data, args.model, args.output)