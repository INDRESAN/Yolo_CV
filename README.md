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

Project Structure 
```
📂 yolo-face-recognition
│── 📂 models              # Pre-trained YOLO model weights
│── 📂 images              # Sample images for testing
│── 📂 videos              # Sample videos for testing
│── detect_faces.py        # Main script for detection
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation
```

How to Use

1️⃣ Download YOLOv8 Model
Download the pre-trained YOLO model weights from the Ultralytics repository:  
```bash
wget https://github.com/ultralytics/assets/releases/download/v8/yolov8n.pt -P models/
```

2️⃣ Run Face Detection on an Image  
```bash
python detect_faces.py --image images/sample.jpg
```

3️⃣ Run Face Detection on a Video
```bash
python detect_faces.py --video videos/sample.mp4
```

### **4️⃣ Run Face Detection in Webcam**  
```bash
python detect_faces.py --webcam
```

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
