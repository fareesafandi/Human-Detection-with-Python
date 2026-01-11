import cv2
import os
import joblib
import kagglehub
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

# 1. Download Dataset
print("Downloading dataset...")

path = kagglehub.dataset_download("constantinwerner/human-detection-dataset")
base_path = os.path.join(path, "human detection dataset")

# 2. Setup HOG Parameters
hog = cv2.HOGDescriptor((64,128), (32,32), (16,16), (32,32), 9)

def prepare_data(limit=300):
    data, labels = [], []
    for label in ['0', '1']: # 0 = No Human, 1 = Human
        folder = os.path.join(base_path, label)
        print(f"Processing folder: {label}...")
        files = os.listdir(folder)
        for i, filename in enumerate(files):
            if i >= limit: break
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (64, 128))
                features = hog.compute(resized)
                data.append(features.flatten())
                labels.append(int(label))
    return np.array(data), np.array(labels)

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# 3. Training Logic
X, y = prepare_data()

print("Training Model 1: SVM...")
model_svm = LinearSVC(max_iter=2000).fit(X, y)
joblib.dump(model_svm, 'models/hog_svm.pkl')

print("Training Model 2: Decision Tree...")
model_tree = DecisionTreeClassifier().fit(X, y)
joblib.dump(model_tree, 'models/decision_tree.pkl')

print("Training Model 3: SGD...")
model_sgd = SGDClassifier().fit(X, y)
joblib.dump(model_sgd, 'models/sgd_model.pkl')

print("\nDONE! All 3 models saved in the /models folder.")