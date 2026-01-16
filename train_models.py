import cv2
import os
import joblib
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

# 1. Download Dataset
print("Downloading dataset...")

path = kagglehub.dataset_download("constantinwerner/human-detection-dataset")
base_path = os.path.join(path, "human detection dataset")

# 2. Setup HOG Parameters
# HOG initialization
winSize = (64,128)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64

hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

def prepare_data(limit=300):

    data, labels = [], []

    print("Loading and augmenting data (this may take a moment)...")
    
    for label in ['0', '1']: # 0 = No Human, 1 = Human
        folder = os.path.join(base_path, label)
        print("In path: ", base_path)
        print(f"Processing folder: {label}...")
        files = os.listdir(folder)
        
        count = 0
        for filename in files:

            if count >= limit: 
                break

            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # 1. Basic Preprocessing
            grayed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(grayed_img, (64, 128), interpolation=cv2.INTER_AREA)
            img_eq = cv2.equalizeHist(resized_img)
            img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)

            # 2. AUGMENTATION LIST
            images_to_process = [img_blur]
            images_to_process.append(cv2.flip(img_blur, 1))
            images_to_process.append(cv2.convertScaleAbs(img_blur, alpha=1.2, beta=10))
            images_to_process.append(cv2.convertScaleAbs(img_blur, alpha=0.8, beta=-10))

            rows, cols = img_blur.shape
            M_left = cv2.getRotationMatrix2D((cols/2, rows/2), 5, 1)
            M_right = cv2.getRotationMatrix2D((cols/2, rows/2), -5, 1)
            images_to_process.append(cv2.warpAffine(img_blur, M_left, (cols, rows)))
            images_to_process.append(cv2.warpAffine(img_blur, M_right, (cols, rows)))

            # 3. Compute Features
            for processed_img in images_to_process:
                data.append(hog.compute(processed_img).flatten())
                labels.append(int(label))
            
            count += 1
        
        print(f"{label} loaded. Current data size: {len(data)}")

    return np.array(data), np.array(labels)

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# 3. Training Logic
X, y = prepare_data()

print(X)
print(y)

# Data Verification
x_df = pd.DataFrame(X)
y_series = pd.Series(y)

print(f"\nFinal Dataset Size: {len(X)}")
print("--- Class Balance ---")
print(y_series.value_counts())

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain set size: {len(x_train)}")
print(f"Test set size: {len(x_test)}")

print("\nTraining Model 1: SVM...")
model_svm = LinearSVC(max_iter=2000).fit(x_train, y_train)
joblib.dump(model_svm, 'models/hog_svm.pkl')

print("Training Model 2: Decision Tree...")
model_tree = DecisionTreeClassifier().fit(x_train, y_train)
joblib.dump(model_tree, 'models/decision_tree.pkl')

print("Training Model 3: Random Forest...")
model_rfc = RandomForestClassifier().fit(x_train, y_train)
joblib.dump(model_rfc, 'models/random_forest.pkl')

print("\nDONE! All 3 models saved in the /models folder.")



models = {'Linear SVM': model_svm, 'Decision Tree': model_tree, 'Random Forest':model_rfc}

def evaluate_model(): 

    results_dict = {}

    for name, model in models.items(): 

        print(f"\n{name}")
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)

        # 2. Store scores in the dictionary
        results_dict[name] = {
            'Accuracy': accuracy,
            'F1': f1score
        }

        print("Accuracy:", accuracy)
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

    return results_dict


def comparison_bar_graph(results_dict):
    model_names = list(results_dict.keys())
    accuracy_scores = [results_dict[m]['Accuracy'] for m in model_names]
    f1_scores = [results_dict[m]['F1'] for m in model_names]

    x = np.arange(len(model_names))  # Label locations
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create the bars
    rects1 = ax.bar(x - width/2, accuracy_scores, width, label='Accuracy', color='#2ecc71')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', color='#3498db')

    # Formatting
    ax.set_ylabel('Score Value')
    ax.set_title('Model Evaluation Comparison: Accuracy vs F1-Score')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Function to add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    
    # Save the chart as a picture for your report
    plt.savefig('evaluation_comparison.png')
    print("Graph saved as 'evaluation_comparison.png'")
    plt.show()


result = evaluate_model()
comparison_bar_graph(result)