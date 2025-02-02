import cv2
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tensorflow as tf
import os
import datetime

# Load your pre-trained model
model = tf.keras.models.load_model('data/model_100_epochs_20000_faces.h5')

# Load OpenCV's pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Group The detected Expressions
positive = [3, 6]
negative = [5, 0, 1, 2]
neutral = [4]

# Function to preprocess the face for the model
def preprocess_face(face):
    # resized_face = cv2.resize(face, (48, 48))  # Adjust size based on model input
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray_face, (48, 48)), -1), 0)
    # gray_face = gray_face / 255.0
    # gray_face = np.expand_dims(gray_face, axis=0)
    # gray_face = np.expand_dims(gray_face, axis=-1)
    return cropped_img

# Main application class
class ExpressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Deteksi Expresi Secara Real-Time")
        self.root.geometry("1200x700")
        
        # Create video capture variable
        self.cap = None
        self.camera_id = 0
        self.is_paused = False  # Flag to pause/resume camera feed
        
        # Initialize counters for expression categories
        self.expression_counts = {'neutral': 0, 'positive': 0, 'negative': 0}
        
         # Create main frames for layout
        self.create_layout()

        # Set initial camera
        self.set_camera(self.camera_id)
        
        # Start video stream update
        self.update_video()
        self.update_graph()

    def create_layout(self):
        """Create the layout of the application."""
        # Frame for video feed (left side)
        video_frame = tk.Frame(self.root)
        video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  # Make it fill the left side

        self.video_label = tk.Label(video_frame, text="Video Feed", font=("Helvetica", 12))
        self.video_label.pack(fill="both", expand=True)  # Make the video feed resizable

        # Frame for buttons and options (bottom of video feed)
        button_frame = tk.Frame(self.root)
        button_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        # Switch Camera Button
        self.switch_camera_btn = ttk.Button(button_frame, text="Ubah Kamera", command=self.switch_camera)
        self.switch_camera_btn.pack(side=tk.LEFT, padx=20, pady=10, ipadx=10, ipady=10, expand=True)

        # Review Statistics Button
        self.pause_btn = ttk.Button(button_frame, text="Pause Kamera", command=self.pause_camera)
        self.pause_btn.pack(side=tk.LEFT, padx=20, pady=10, ipadx=10, ipady=10, expand=True)

        # Frame for expression graph (right side)
        graph_frame = tk.Frame(self.root)
        graph_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")  # Fill the right side

        # Make sure the right column (graph) expands with window resize
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Matplotlib graph to show expression counts
        self.fig, self.ax = plt.subplots(figsize=(5, 4))  # Slightly larger plot
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.get_tk_widget().pack()

    def set_camera(self, camera_id):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            self.cap = None
            messagebox.showerror("Error", "Kamera Tidak Dapat Diakses")
        else:
            self.camera_id = camera_id
    
    def switch_camera(self):
        self.camera_id = (self.camera_id + 1) % 5  # Assume max 5 cameras
        self.set_camera(self.camera_id)

    def detect_faces_and_expressions(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            processed_face = preprocess_face(face)
            prediction = model.predict(processed_face)
            predicted_label = np.argmax(prediction)
            # predicted_label = 2

            if predicted_label in neutral:
                expression = 'neutral'
                self.expression_counts['neutral'] += 1
            elif predicted_label in positive:
                expression = 'positive'
                self.expression_counts['positive'] += 1
            else:
                expression = 'negative'
                self.expression_counts['negative'] += 1

            # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # cv2.putText(frame, expression, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return frame

    def update_video(self):
        if not self.is_paused:
            if self.cap:
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.resize(frame, (400, 300))
                    frame_with_faces = self.detect_faces_and_expressions(frame)
                    frame_rgb = cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_video)

    def update_graph(self):
        self.ax.clear()
        categories = ['Netral', 'Positif', 'Negatif']
        counts = [self.expression_counts['neutral'], self.expression_counts['positive'], self.expression_counts['negative']]
        self.ax.bar(categories, counts, color=['gray', 'green', 'red'])
        self.ax.set_title('Jumlah Expresi')
        self.canvas.draw()
        self.root.after(1000, self.update_graph)

    def pause_camera(self):
        """Pause the camera feed and show statistics in a new window."""
        self.is_paused = True

        # Create a new window to display the statistics
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Hasil Deteksi Sementara")
        stats_window.geometry("400x300")

        # Display the counts of each expression
        neutral_count = tk.Label(stats_window, text=f"Netral: {self.expression_counts['neutral']}")
        neutral_count.pack(pady=10)

        positive_count = tk.Label(stats_window, text=f"Postif: {self.expression_counts['positive']}")
        positive_count.pack(pady=10)

        negative_count = tk.Label(stats_window, text=f"Negatif: {self.expression_counts['negative']}")
        negative_count.pack(pady=10)

        # Add a button to close the statistics window and resume the camera feed
        close_button = tk.Button(stats_window, text="Lanjutkan Kamera", command=lambda: self.resume_camera(stats_window))
        close_button.pack(padx=20, pady=10, ipadx=10, ipady=10)

    def resume_camera(self, stats_window):
        """Resume the camera feed and close the statistics window."""
        self.is_paused = False
        stats_window.destroy()


    def generate_report(self):
        """Generate a report with expression counts and save the graph image when the app is closed."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"expression_report_{timestamp}.txt"
        graph_filename = f"expression_graph_{timestamp}.png"

        # Save the final graph as an image
        categories = ['Netral', 'Positif', 'Negatif']
        counts = [self.expression_counts['neutral'], self.expression_counts['positive'], self.expression_counts['negative']]
    
        # Plot and save the image
        fig, ax = plt.subplots()
        ax.bar(categories, counts, color=['gray', 'green', 'red'])
        ax.set_title('Jumlah Ekspresi')
        ax.set_xlabel('Ekspresi')
        ax.set_ylabel('Jumlah')
        plt.savefig(graph_filename)
        plt.close(fig)

        # Write the report text file
        report_content = (
            f"Report Hasil Deteksi\n"
            f"===========================\n\n"
            f"Tanggal & Waktu: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Ekspresi Yang Dideteksi:\n"
            f" - Netral: {self.expression_counts['neutral']}\n"
            f" - Positif: {self.expression_counts['positive']}\n"
            f" - Negatif: {self.expression_counts['negative']}\n\n"
            f"Gambar Grafik Disimpan pada File : {graph_filename}\n"
        )

        with open(report_filename, "w") as file:
            file.write(report_content)
        
        messagebox.showinfo("Report Berhasil Disimpan", f"Report Disimpan dengan nama {report_filename} dan grafik disimpan sebagai {graph_filename}")


    def on_close(self):
        if self.cap:
            self.cap.release()

        self.generate_report()  # Generate the report upon closing
        self.root.quit()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ExpressionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
