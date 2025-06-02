import streamlit as st
import cv2
import numpy as np
from face_encoder import FaceEncoder
from PIL import Image
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Face Encoder", layout="wide")

st.title("Face Encoder - CNN Feature Extraction")
st.write("Upload an image to extract face embeddings using CNN.")

# Main content
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Read and process image
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Initialize encoder
        encoder = FaceEncoder()

        # Detect faces
        face_images = encoder.detect_face(image_bgr)
        faces = encoder.face_detector.detect_faces(image_np)

        if faces:
            # Draw bounding boxes
            for face in faces:
                x, y, w, h = face['box']
                cv2.rectangle(image_np, (x, y), (x+w, y+h), (0,255,0), 2)
            st.image(image_np, caption=f"Detected {len(faces)} Faces", use_container_width=True)

            # Get embeddings
            embeddings = encoder.encode_face(image_bgr)
            if embeddings is not None:
                st.success(f"Successfully extracted embeddings for {len(embeddings)} faces!")
            else:
                st.warning("Faces detected, but embeddings could not be extracted.")
        else:
            st.image(image_np, caption="No face detected.", use_container_width=True)
            st.warning("No face detected in the image.")

    with col2:
        if faces and embeddings is not None:
            st.subheader("Face Embeddings Analysis")
            
            # Create tabs for each face
            face_tabs = st.tabs([f"Face {i+1}" for i in range(len(embeddings))])
            
            for i, (face_tab, embedding) in enumerate(zip(face_tabs, embeddings)):
                with face_tab:
                    st.write(f"**Analysis for Face {i+1}**")
                    
                    # Create a DataFrame for the embedding vector
                    embedding_df = pd.DataFrame({
                        'Dimension': range(1, len(embedding) + 1),
                        'Value': embedding
                    })
                    
                    # Display embedding statistics
                    st.write("**Embedding Statistics:**")
                    stats_df = pd.DataFrame({
                        'Metric': ['Mean', 'Std Dev', 'Min', 'Max'],
                        'Value': [
                            np.mean(embedding),
                            np.std(embedding),
                            np.min(embedding),
                            np.max(embedding)
                        ]
                    })
                    st.table(stats_df)
                    
                    # Visualize embedding vector
                    st.write("**Embedding Vector Visualization:**")
                    fig = go.Figure(data=go.Scatter(
                        y=embedding,
                        mode='lines+markers',
                        name='Embedding Values'
                    ))
                    fig.update_layout(
                        title=f'128-Dimensional Face Embedding - Face {i+1}',
                        xaxis_title='Dimension',
                        yaxis_title='Value',
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display embedding vector in a table
                    st.write("**Raw Embedding Vector:**")
                    st.dataframe(embedding_df.style.format({'Value': '{:.4f}'}))
            
            # Additional information
            st.info("""
            **About the Embeddings:**
            - 128-dimensional vectors representing facial features
            - Values are L2-normalized
            - Can be used for face comparison and recognition
            - Each face has its own unique embedding
            """) 