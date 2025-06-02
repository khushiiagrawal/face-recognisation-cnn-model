# Face Recognition CNN Model

Face recognition system built using CNN (Convolutional Neural Network) and Streamlit for an interactive user interface. The system uses InceptionV3 as the base model for feature extraction and MTCNN for face detection.

## Features
-  Streamlit interactive tabbed interface 
- Multi-face detection and recognition 
- CNN-based face encoding 
- Detailed face embedding analysis and visualization
- L2-normalized 128-dimensional face embeddings

## Technical Details
- Face Detection: MTCNN (Multi-task Cascaded Convolutional Networks)
- Base Model: InceptionV3 with ImageNet weights
- Embedding Dimension: 128
- Input Image Size: 160x160x3
- Face Embedding Features:
  - L2-normalized vectors
  - Statistical analysis (mean, std dev, min, max)
  - Interactive visualization
  - Raw embedding data display

## Prerequisites
- Python 3.11 
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/khushiiagrawal/face-recognition-cnn-model.git
cd face-recognition-cnn-model
```

2. Create and activate a virtual environment (3.11 recommended):
```bash
python3.11 -m venv venv
source venv/bin/activate # For macos

# On Windows, use: 
venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run streamlit_app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (http://localhost:8501)

3. Upload an image containing one or more faces

4. The application will:
   - Detect all faces in the image
   - Display the image with face bounding boxes
   - Generate embeddings for each detected face
   - Show detailed analysis for each face in separate tabs
   - Provide statistical information and visualizations

