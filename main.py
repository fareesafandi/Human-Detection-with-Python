import tkinter as tk
from tkinter import ttk
from tkinter import ttk, filedialog, messagebox
import cv2
import joblib
import numpy as np
from PIL import Image, ImageTk

class HumanDetectionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Multi-Model Human Detector")

        # Initialize Webcam
        self.cap = cv2.VideoCapture(0)
        self.hog = cv2.HOGDescriptor((64,128), (32,32), (16,16), (32,32), 9)

        # UI Elements
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        # Model Selector Dropdown
        self.model_label = tk.Label(window, text="Select Model:")
        self.model_label.pack(side="left", padx=10)
        
        self.model_choice = tk.StringVar(value="hog_svm")
        self.dropdown = ttk.Combobox(window, textvariable=self.model_choice)
        self.dropdown['values'] = ("hog_svm", "decision_tree", "sgd_model")
        self.dropdown.pack(side="left")

        self.btn_predict = tk.Button(window, text="CAPTURE & PREDICT", command=self.predict, bg="green", fg="white", height=2)
        self.btn_predict.pack(side="right", fill="x", expand=True)

        self.result_label = tk.Label(window, text="Result: Waiting...", font=("Arial", 18, "bold"))
        self.result_label.pack(side="bottom", pady=10)

        self.update_webcam()
        self.window.mainloop()

    def update_webcam(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            # Convert BGR to RGB for Tkinter
            cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2_img))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        self.window.after(15, self.update_webcam)

    def predict(self):
        try:
            # 1. Load the selected model
            model_path = f"models/{self.model_choice.get()}.pkl"
            model = joblib.load(model_path)

            # 2. Process current frame
            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 128))
            features = self.hog.compute(resized).reshape(1, -1)

            # 3. Predict
            prediction = model.predict(features)
            
            # 4. Update UI
            if prediction[0] == 1:
                self.result_label.config(text="Result: HUMAN DETECTED", fg="red")
            else:
                self.result_label.config(text="Result: EMPTY / NO HUMAN", fg="blue")
        
        except Exception as e:
            self.result_label.config(text="Error: Train models first!", fg="orange")

# Run App
if __name__ == "__main__":
    root = tk.Tk()
    app = HumanDetectionApp(root)