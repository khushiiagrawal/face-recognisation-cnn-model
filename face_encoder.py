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
        base_model = tf.keras.applications.InceptionV3(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape
        )
        base_model.trainable = False

        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation=None)(x)
        x = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)

        return Model(inputs=base_model.input, outputs=x)

    def detect_faces(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.face_detector.detect_faces(rgb_image)
        face_images = []

        for face in faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)  # prevent negative indices
            face_img = image[y:y+h, x:x+w]
            if face_img.size:
                face_images.append((face_img, (x, y, w, h)))
        
        return face_images

    def preprocess_face(self, face_img):
        face_img = cv2.resize(face_img, (self.input_shape[0], self.input_shape[1]))
        face_img = face_img.astype('float32') / 255.0
        return np.expand_dims(face_img, axis=0)

    def encode_faces(self, image):
        face_data = self.detect_faces(image)
        embeddings = []

        for face_img, _ in face_data:
            processed = self.preprocess_face(face_img)
            embedding = self.model.predict(processed, verbose=0)
            embeddings.append(embedding[0])
        
        return embeddings, [box for _, box in face_data]
