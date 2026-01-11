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
hog = cv2.HOGDescriptor((64,128), (32,32), (16,16), (32,32), 9)

def prepare_data(limit=300):

    data, labels = [], []

    for label in ['0', '1']: # 0 = No Human, 1 = Human
        folder = os.path.join(base_path, label)
        print("In path: ", base_path)
        print(f"Processing folder: {label}...")
        files = os.listdir(folder)

        for i, filename in enumerate(files):

            if i >= limit: break

            img = cv2.imread(os.path.join(folder, filename))

            if img is not None:

                # Data Preprocessing
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

print(X)
print(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



print("Training Model 1: SVM...")
model_svm = LinearSVC(max_iter=2000).fit(X, y)
joblib.dump(model_svm, 'models/hog_svm.pkl')

print("Training Model 2: Decision Tree...")
model_tree = DecisionTreeClassifier().fit(X, y)
joblib.dump(model_tree, 'models/decision_tree.pkl')

print("Training Model 3: Random Forest...")
model_rfc = RandomForestClassifier().fit(X, y)
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
    """
    Generates a bar graph comparing Accuracy and F1-Score.
    results_dict should look like: 
    {'SVM': {'Accuracy': 0.88, 'F1': 0.85}, 'Decision Tree': {...}, 'Random Forest': {...}}
    """
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
