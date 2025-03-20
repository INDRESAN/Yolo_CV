YOLO Face Recognition using OpenCV  

Overview
This project uses the **YOLO (You Only Look Once)** object detection algorithm along with **OpenCV** to detect and recognize faces in real-time from images, videos, or webcam feeds. YOLO is a deep learning-based object detection model that can perform fast and accurate face recognition.  

Features  
✅ Real-time face detection using YOLOv8  
✅ Supports image, video, and webcam input  
✅ Uses OpenCV for processing and display  
✅ Fast and efficient detection  

Requirements  
Make sure you have the following dependencies installed before running the project.  

Install Dependencies  
```bash
pip install ultralytics opencv-python numpy pyyaml
```



How to Use

1️⃣ Set Up Label Studio for Annotation
1. Open **Anaconda Prompt** and navigate to Label Studio:
   ```bash
   conda activate your_env_name  # Activate your Conda environment
   label-studio
   ```
2. Upload your dataset (images containing faces).
3. Annotate the images with the appropriate **face recognition classes**.
4. Export the labeled dataset in **YOLO format** (including `.txt` annotation files).
5. Save the exported dataset in the `datasets/` folder.

---

2️⃣ Train the YOLO Model in Google Colab
1. Open **Google Colab** and upload the **annotated dataset**.
2. Run `yolo_cv.py` to train the model:
   ```bash
   python yolo_cv.py --data datasets/ --epochs 50 --output my_model.pt
   ```
3. This script will train YOLO and generate the **trained model file (`my_model.pt`)**.

---

3️⃣ Execute YOLO Face Recognition Using Anaconda
1. Move the `my_model.pt` file to the **YOLO detection folder**.
2. Open **Anaconda Prompt** and navigate to the detection script:
   ```bash
   cd path/to/yolo_detection
   conda activate your_env_name
   ```
3. Run the **YOLO detection script (`yolo_detect.py`)**:
   ```bash
   python yolo_detect.py --model my_model.pt
   ```
4. The script will:
   - Open the webcam.
   - Recognize **faces based on the trained model**.
   - Display the live detection results.
   - 
Final Notes
- Label Studio is used for annotating images.
- Colab is used for training YOLO with the dataset.
- Anaconda + PyTorch + OpenCV is used for running the detection model on a local machine.

Code: detect_faces.py
```python
import cv2
import torch
from ultralytics import YOLO

# Load YOLO model
model = YOLO("models/yolov8n.pt")  

# Load image
image = cv2.imread("images/sample.jpg")

# Run inference
results = model(image)

# Draw bounding boxes
for result in results:
    for box in result.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display image
cv2.imshow("YOLO Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Future Improvements
🔹 Train a custom YOLO model for better accuracy  
🔹 Add face recognition (match detected faces with a database)  
🔹 Improve real-time performance for large-scale applications  

Credits  
- YOLOv8 by [Ultralytics](https://github.com/ultralytics/ultralytics)  
- OpenCV for image processing  
- Edje Electronics
