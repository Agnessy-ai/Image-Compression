import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Image Compression with SVD", layout="wide")

st.title("🖼️ Image Compression using SVD")
st.markdown("Upload an image and choose how many singular values (k) to keep. Higher `k` = better quality but less compression.")

# Upload
uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

def compress_channel(channel_data, k):
    U, S, VT = np.linalg.svd(channel_data, full_matrices=False)
    S_k = np.diag(S[:k])
    compressed = np.dot(U[:, :k], np.dot(S_k, VT[:k, :]))
    return np.clip(compressed, 0, 255)

def compress_image_svd(image, k):
    img_array = np.array(image, dtype=float)
    if len(img_array.shape) == 2:  # Grayscale
        compressed = compress_channel(img_array, k)
        return Image.fromarray(compressed.astype(np.uint8))
    else:
        compressed_channels = [
            compress_channel(img_array[:, :, i], k) for i in range(3)
        ]
        compressed_rgb = np.stack(compressed_channels, axis=2)
        return Image.fromarray(compressed_rgb.astype(np.uint8))

# If image is uploaded
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    max_k = min(image.size)  # Limit slider to image size
    k = st.slider("Select number of singular values (k)", 5, max_k, value=50, step=5)

    # Compress
    with st.spinner("Compressing..."):
        compressed_image = compress_image_svd(image, k)

    # Show comparison
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_column_width=True)
    with col2:
        st.subheader(f"Compressed (k = {k})")
        st.image(compressed_image, use_column_width=True)

    # Download
    buf = io.BytesIO()
    compressed_image.save(buf, format="PNG")
    st.download_button("📥 Download Compressed Image", buf.getvalue(), file_name=f"compressed_k{k}.png", mime="image/png")
