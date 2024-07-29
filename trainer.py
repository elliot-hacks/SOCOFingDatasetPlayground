import os
import cv2
import pickle
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def extract_features(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logging.error(f"Image at path {image_path} could not be read.")
        return None

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    if descriptors is None:
        logging.error(f"No descriptors found for image at path {image_path}.")
        return None

    # Flatten the descriptors to use as features
    features = descriptors.flatten()

    # Normalize the length of the feature vector
    max_features = 500  # Example: take the first 500 features
    if len(features) > max_features:
        features = features[:max_features]
    else:
        # Calculate the mean of the feature vector and use that as the padding value
        mean_feature = np.mean(features)
        features = np.pad(features, (0, max_features - len(features)), 'constant', constant_values=mean_feature)

    return features

def load_data(directory: str) -> (np.ndarray, np.ndarray):
    X = []
    y = []

    for label in os.listdir(directory):
        person_dir = os.path.join(directory, label)
        if not os.path.isdir(person_dir):
            continue

        for file in os.listdir(person_dir):
            file_path = os.path.join(person_dir, file)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(label)

    # if not X or not y:
    #     raise ValueError("No valid samples found. Ensure the directory contains valid images.")

    X = np.array(X)
    y = np.array(y)

    return X, y

def train_model(X: np.ndarray, y: np.ndarray, n_neighbors: int = 3) -> KNeighborsClassifier:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X, y)
    return knn

def save_model(model: KNeighborsClassifier, filename: str) -> None:
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def main() -> None:
    directory = "Real/"
    X, y = load_data(directory)
    model = train_model(X, y)
    save_model(model, 'knn_model.pkl')
    logging.info(f"Model trained and saved as 'knn_model.pkl' with {len(X)} samples.")

if __name__ == '__main__':
    main()
