import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import joblib
import numpy as np
from PIL import Image, ImageTk

class HumanDetectionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Multi-Model Human Detector")

        # Initialize Webcam and HOG
        self.cap = cv2.VideoCapture(0)
        
        # HOG initialization with updated parameters
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
        
        self.hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                                    histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
        
        # State variables
        self.is_using_webcam = True
        self.current_frame = None

        # --- UI Elements ---
        self.canvas = tk.Canvas(window, width=640, height=480, bg="black")
        self.canvas.pack(pady=5)

        # Button Frame (Top)
        self.top_frame = tk.Frame(window)
        self.top_frame.pack(pady=5)

        self.btn_upload = tk.Button(self.top_frame, text="📁 Upload Picture", command=self.upload_picture)
        self.btn_upload.pack(side="left", padx=5)

        self.btn_webcam = tk.Button(self.top_frame, text="🎥 Live Webcam", command=self.use_webcam)
        self.btn_webcam.pack(side="left", padx=5)

        # Model Selector and Predict Frame
        self.mid_frame = tk.Frame(window)
        self.mid_frame.pack(pady=10)

        tk.Label(self.mid_frame, text="Select Model:").pack(side="left", padx=5)
        self.model_choice = tk.StringVar(value="hog_svm")
        self.dropdown = ttk.Combobox(self.mid_frame, textvariable=self.model_choice)
        self.dropdown['values'] = ("hog_svm", "decision_tree", "random_forest")
        self.dropdown.pack(side="left", padx=5)

        self.btn_predict = tk.Button(self.mid_frame, text="RUN PREDICTION", command=self.predict, 
                                   bg="green", fg="white", font=("Arial", 10, "bold"))
        self.btn_predict.pack(side="left", padx=10)

        # Result Label
        self.result_label = tk.Label(window, text="Result: Waiting...", font=("Arial", 18, "bold"))
        self.result_label.pack(side="bottom", pady=10)

        self.update_loop()
        self.window.mainloop()

    def update_loop(self):
        """Webcam loop: only updates if in webcam mode."""
        if self.is_using_webcam:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.display_image(frame)
        
        self.window.after(15, self.update_loop)

    def display_image(self, frame):
        """Converts CV2 frame to Tkinter format and displays it."""
        cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2_img = cv2.resize(cv2_img, (640, 480))
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2_img))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def upload_picture(self):
        """Stops webcam and allows user to pick a file."""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                self.is_using_webcam = False
                self.current_frame = image
                self.display_image(image)
                self.result_label.config(text="Image Loaded", fg="black")

    def use_webcam(self):
        """Switches back to webcam mode."""
        self.is_using_webcam = True
        self.result_label.config(text="Live Webcam Mode", fg="black")

    def predict(self):
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No image source found!")
            return

        try:
            # 1. Load the selected model
            model_path = f"models/{self.model_choice.get()}.pkl"
            model = joblib.load(model_path)

            # 2. Process current frame with updated preprocessing
            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 128), interpolation=cv2.INTER_AREA)
            img_eq = cv2.equalizeHist(resized)
            img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)
            features = self.hog.compute(img_blur).reshape(1, -1)

            # 3. Predict
            prediction = model.predict(features)
            
            # 4. Update UI
            if prediction[0] == 1:
                self.result_label.config(text="Result: HUMAN DETECTED", fg="red")
            else:
                self.result_label.config(text="Result: EMPTY / NO HUMAN", fg="blue")
        
        except Exception as e:
            self.result_label.config(text="Error: Model missing!", fg="orange")
            print(f"Error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = HumanDetectionApp(root)