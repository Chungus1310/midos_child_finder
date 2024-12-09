# **Midos Child Finder**

Midos Child Finder is a video processing tool designed to detect and classify faces in a video as **"Child"** or **"Adult"**. This repository leverages **YOLOv8** for face detection and a **FastAI-based classifier** for distinguishing between children and adults. The project includes tools for model training and video processing, ensuring a streamlined workflow.

---

## **Table of Contents**
1. [Project Structure](#project-structure)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
    - [Model Training](#model-training)
    - [Video Processing](#video-processing)
5. [Input and Output Details](#input-and-output-details)
6. [Dependencies](#dependencies)
7. [How It Works](#how-it-works)
8. [License](#license)

---

## **Project Structure**
```plaintext
midos_child_finder/
├── children_vs_adults_model.pkl   # Trained FastAI classification model
├── yolov8n-face.pt               # YOLOv8 model for face detection
├── train/
│   ├── train.ipynb               # Notebook for training the classification model
│   ├── data/                     # Folder to store training images (if applicable)
│   └── ...                       # Any additional files for training
├── process_video/
│   ├── video.ipynb               # Notebook for processing videos
│   └── ...                       # Related helper scripts (if added)
└── README.md                     # Project documentation (this file)
```

**Dataset Used**:
```
https://www.kaggle.com/datasets/die9origephit/children-vs-adults-images
```


---

## **Features**
- **Face Detection**:
  - Uses **YOLOv8** for robust and efficient face detection.
- **Classification**:
  - Classifies detected faces as "Child" or "Adult" using a trained **FastAI model**.
- **Video Processing**:
  - Processes videos frame-by-frame.
  - Annotates faces with bounding boxes and classification labels.
  - Retains original audio for the output video.
- **GPU Acceleration**:
  - Fully utilizes GPU resources for faster processing.

---

## **Installation**

1. Clone this repository:
    ```bash
    git clone https://github.com/Chungus1310/midos_child_finder.git
    cd midos_child_finder
    ```

2. Install required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```
    **Note**: Ensure you have a GPU-compatible setup for optimal performance (e.g., CUDA installed).

3. Install **FFmpeg**:
    - On Linux:
        ```bash
        sudo apt update
        sudo apt install ffmpeg
        ```
    - On MacOS:
        ```bash
        brew install ffmpeg
        ```
    - On Windows:
        - Download FFmpeg from [FFmpeg's official site](https://ffmpeg.org/), and add it to your PATH.

---

## **Usage**

### **Model Training**
1. Navigate to the `train` folder:
    ```bash
    cd train
    ```
2. Open the `train.ipynb` notebook:
    ```bash
    jupyter notebook train.ipynb
    ```
3. Follow the instructions in the notebook to train the classification model.
    - Ensure your dataset (images of children and adults) is stored in the `data` directory.
    - The output trained model will be saved as `children_vs_adults_model.pkl` in the repository's root directory.

---

### **Video Processing**
1. Navigate to the `process_video` folder:
    ```bash
    cd process_video
    ```
2. Open the `video.ipynb` notebook:
    ```bash
    jupyter notebook video.ipynb
    ```
3. Follow the instructions in the notebook to process your video:
    - Ensure `yolov8n-face.pt` and `children_vs_adults_model.pkl` are in the repository's root directory.
    - Specify the input video path and desired output path.
    - Run the notebook to annotate the video.

4. **Output**:
    - The annotated video will be saved with bounding boxes and labels for each detected face.

---

## **Input and Output Details**
- **Input Video**:
  - Any `.mp4` file.
  - Ensure the video resolution and frame rate are suitable for processing (high resolutions may increase runtime).
- **Output Video**:
  - Annotated `.mp4` video with bounding boxes and classification labels for each face.
  - Original audio is preserved.

---

## **Dependencies**
This project requires the following libraries and tools:
- **Python 3.8+**
- **Libraries**:
    - `opencv-python-headless`
    - `fastai`
    - `ultralytics`
    - `ffmpeg-python`
    - `torch` (for GPU acceleration)
- **FFmpeg**:
    - Required for audio and video stream handling.



---

## **How It Works**

1. **Face Detection**:
    - The `yolov8n-face.pt` model is used to detect faces in each frame of the video.
    - Detected faces are cropped and passed to the classifier.

2. **Classification**:
    - The cropped face is resized to 224x224 and classified using the `children_vs_adults_model.pkl` trained with FastAI.
    - Faces are labeled as **"Child"** or **"Adult"**, along with a confidence score.

3. **Annotation**:
    - Bounding boxes are drawn around faces in the frame.
    - Labels and confidence scores are displayed above each box.

4. **Video Processing**:
    - Each frame is processed and written to a temporary video file.
    - FFmpeg combines the annotated video frames with the original audio.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgements**
- **YOLOv8**: For efficient face detection.
- **FastAI**: For easy model training and deployment.
- **FFmpeg**: For handling audio and video streams seamlessly.

---

Feel free to reach out via GitHub issues for questions or suggestions. Happy coding! 🎉
