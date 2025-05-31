import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import cv2
from mtcnn import MTCNN

class FaceEncoder:
    def __init__(self, input_shape=(160, 160, 3)):
        self.input_shape = input_shape
        self.model = self._build_model()
        self.face_detector = MTCNN()
        
    def _build_model(self):
        """Build the CNN model for face encoding"""
        base_model = tf.keras.applications.InceptionV3(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        # Add custom layers for face encoding
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation=None)(x)  # No activation for embedding layer
        x = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)  # L2 normalization
        
        model = Model(inputs=base_model.input, outputs=x)
        return model
    
    def preprocess_face(self, face_img):
        """Preprocess face image for the model"""
        # Resize to model input size
        face_img = cv2.resize(face_img, (self.input_shape[0], self.input_shape[1]))
        # Convert to RGB if needed
        if len(face_img.shape) == 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        # Normalize pixel values
        face_img = face_img.astype('float32') / 255.0
        # Add batch dimension
        face_img = np.expand_dims(face_img, axis=0)
        return face_img
    
    def detect_face(self, image):
        """Detect face in the image using MTCNN"""
        # Convert image to RGB if it's BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
            
        # Detect faces
        faces = self.face_detector.detect_faces(rgb_image)
        
        if not faces:
            return None
            
        # Get the first face
        face = faces[0]
        x, y, width, height = face['box']
        
        # Extract face region
        face_img = image[y:y+height, x:x+width]
        return face_img
    
    def encode_face(self, image):
        """Extract face embedding from an image"""
        # Detect face
        face_img = self.detect_face(image)
        if face_img is None:
            return None
            
        # Preprocess face
        processed_face = self.preprocess_face(face_img)
        
        # Get face embedding
        embedding = self.model.predict(processed_face)
        return embedding[0]  # Return the first (and only) embedding
    
    def compare_faces(self, embedding1, embedding2, threshold=0.6):
        """Compare two face embeddings and return similarity score"""
        distance = np.linalg.norm(embedding1 - embedding2)
        return distance < threshold, distance

# Example usage
if __name__ == "__main__":
    # Initialize face encoder
    encoder = FaceEncoder()
    
    # Example: Load and encode a face
    image = cv2.imread("image.jpg")
    if image is not None:
        embedding = encoder.encode_face(image)
        if embedding is not None:
            print("Face embedding shape:", embedding.shape)
            print("Face embedding:", embedding) 