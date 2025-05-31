# Face Encoder

This project implements a CNN-based face encoding system that extracts facial embeddings (128-dimensional vectors) from images. The system uses a pre-trained InceptionV3 model as a base and adds custom layers for face encoding.

## Features

- Face detection using MTCNN
- Face embedding extraction using a CNN model
- Face comparison functionality
- Preprocessing pipeline for face images

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

Here's a basic example of how to use the face encoder:

```python
from face_encoder import FaceEncoder
import cv2

# Initialize the face encoder
encoder = FaceEncoder()

# Load an image
image = cv2.imread("path_to_image.jpg")

# Extract face embedding
embedding = encoder.encode_face(image)

if embedding is not None:
    print("Face embedding shape:", embedding.shape)
    print("Face embedding:", embedding)
```

## Model Architecture

The face encoder uses the following architecture:
1. InceptionV3 base model (pre-trained on ImageNet)
2. Global Average Pooling
3. Dense layer (512 units) with ReLU activation
4. Batch Normalization
5. Dropout (0.5)
6. Dense layer (128 units) for embedding
7. L2 normalization

## Face Comparison

To compare two face embeddings:

```python
# Compare two face embeddings
similar, distance = encoder.compare_faces(embedding1, embedding2, threshold=0.6)
print(f"Faces are similar: {similar}")
print(f"Distance: {distance}")
```

## Notes

- The model expects input images of size 160x160x3
- Face detection is performed using MTCNN
- The embedding dimension is 128
- The comparison threshold is set to 0.6 by default 