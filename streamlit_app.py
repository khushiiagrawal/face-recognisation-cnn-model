import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
from face_encoder import FaceEncoder
import traceback

st.set_page_config(page_title="Face Encoder", layout="wide")
st.title("Face Encoder - CNN Feature Extraction")
st.write("Upload an image to extract face embeddings using CNN.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        col1, col2 = st.columns(2)

        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        encoder = FaceEncoder()
        embeddings, boxes = encoder.encode_faces(image_bgr)

        with col1:
            for box in boxes:
                x, y, w, h = box
                cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if boxes:
                st.image(image_np, caption=f"Detected {len(boxes)} Face(s)", use_container_width=True)
                st.success(f"Successfully extracted {len(embeddings)} face embeddings.")
            else:
                st.image(image_np, caption="No face detected.", use_container_width=True)
                st.warning("No face detected in the image.")

        with col2:
            if boxes:
                tabs = st.tabs([f"Face {i+1}" for i in range(len(embeddings))])

                for i, (tab, emb) in enumerate(zip(tabs, embeddings)):
                    with tab:
                        st.write(f"**Analysis for Face {i+1}**")
                        stats = pd.DataFrame({
                            "Metric": ["Mean", "Std Dev", "Min", "Max"],
                            "Value": [
                                np.mean(emb),
                                np.std(emb),
                                np.min(emb),
                                np.max(emb)
                            ]
                        })
                        st.table(stats)

                        st.write("**Vector Visualization**")
                        fig = go.Figure(go.Scatter(y=emb, mode='lines+markers'))
                        fig.update_layout(title="Face Embedding", xaxis_title="Dimension", yaxis_title="Value")
                        st.plotly_chart(fig, use_container_width=True)

                        st.write("**Raw Embedding Vector**")
                        df = pd.DataFrame({'Dimension': list(range(1, 129)), 'Value': emb})
                        st.dataframe(df.style.format({'Value': '{:.4f}'}))
    except Exception as e:
        st.error("An error occurred while processing the image.")
        st.text(traceback.format_exc())
