import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor

IMG_SIZE = 128
BATCH_SIZE = 32
SUBSET_SIZE = 12000

start_time = time.time()

# Seting up the ResNet50 model with Global Average Pooling
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
model = Model(inputs=base_model.input, outputs=x)

base_model.trainable = False

def load_and_preprocess_single_image(img_path):
    """
    Loads and preprocesses a single image to be fed into the model.

    Args:
    - img_path (str): Path to the image file.

    Returns:
    - np.ndarray: Preprocessed image array ready for model input.
    """
    print(f"Loading and preprocessing image: {img_path}")
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Function to extract features for a batch of images
def extract_image_features_in_batch(image_paths, model, batch_size=BATCH_SIZE):
    """
    Extracts features for a batch of images.

    Args:
    - image_paths (list): List of image file paths.
    - model (keras.Model): Pre-trained model for feature extraction.

    Returns:
    - np.ndarray: Extracted features for each image in the batch.
    """
    with ThreadPoolExecutor() as executor:
        processed_images = list(executor.map(load_and_preprocess_single_image, image_paths))

    batch_images = np.vstack(processed_images)
    print(f"Extracted features for batch of {len(image_paths)} images")
    return model.predict(batch_images).reshape(batch_images.shape[0], -1)

def extract_all_image_features(image_folder, model, subset_size=SUBSET_SIZE, batch_size=BATCH_SIZE):
    """
    Extracts features from all images in the dataset.

    Args:
    - image_folder (str): Path to the folder containing images.
    - model (keras.Model): Pre-trained model for feature extraction.

    Returns:
    - np.ndarray: Extracted features from the dataset.
    - np.ndarray: Encoded labels for the dataset.
    - LabelEncoder: The label encoder used to encode labels.
    """
    features = []
    labels = []
    img_paths = []

    # Collect image paths and labels
    for label in os.listdir(image_folder):
        label_path = os.path.join(image_folder, label)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                if filename.endswith(('.jpg', '.png')):
                    img_paths.append(os.path.join(label_path, filename))
                    labels.append(label)

    img_paths = img_paths[:subset_size]
    labels = labels[:subset_size]


    for start in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[start:start + batch_size]
        batch_features = extract_image_features_in_batch(batch_paths, model, batch_size)
        features.append(batch_features)

    features = np.vstack(features)


    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    print(f"Extracted features and encoded labels for {len(encoded_labels)} images")
    return features, encoded_labels, label_encoder


image_folder = r'C:\Users\isken\PycharmProjects\pythonProject\crop-images\Plant-images'
print("Starting feature extraction process...")
features, labels, label_encoder = extract_all_image_features(image_folder, model)


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)
print(f"Split data into {len(X_train)} training samples and {len(X_test)} testing samples")


model_xgb = xgb.XGBClassifier(
    n_estimators=30,
    max_depth=1,
    learning_rate=0.01,
    reg_alpha=0.1,
    reg_lambda=0.1,
    eval_metric='logloss'
)


eval_set = [(X_train, y_train), (X_test, y_test)]
model_xgb.fit(X_train, y_train, eval_set=eval_set, verbose=True)


y_pred = model_xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


print("Classification Report:")
print(classification_report(y_test, y_pred))


eval_results = model_xgb.evals_result()

plt.figure(figsize=(10, 7))
plt.plot(eval_results['validation_0']['logloss'], label='Train')
plt.plot(eval_results['validation_1']['logloss'], label='Test')
plt.xlabel('Iterations')
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss Over Iterations')
plt.legend()
plt.show()


end_time = time.time()
duration = (end_time - start_time) / 60
print(f"Total time taken: {duration:.2f} minutes")
