import streamlit as st
import numpy as np
import cv2
from descriptor import glcm, bitdesc
from distances import retrieve_similar_images

# Page configuration
st.set_page_config(
    page_title="Image Similarity Search",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #1E3D59;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #17A2B8;
        color: white;
    }
    .upload-prompt {
        text-align: center;
        padding: 2rem;
        border: 2px dashed #cccccc;
        border-radius: 5px;
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load signatures
try:
    signatures = np.load('signatures.npy')
    st.sidebar.success("✅ Signatures database loaded successfully")
except Exception as e:
    st.sidebar.error("❌ Error loading signatures database. Please ensure 'signatures.npy' exists.")
    st.stop()

# Application header
st.title("🔍 Image Similarity Search")
st.markdown("---")

# Sidebar configuration
st.sidebar.title("Search Configuration")
num_images = st.sidebar.slider("Number of similar images", min_value=1, max_value=10, value=5)
distance_measure = st.sidebar.selectbox(
    "Distance Measure",
    ["euclidean", "manhattan", "chebyshev", "canberra"],
    help="Choose the method to calculate similarity between images"
)
descriptor = st.sidebar.selectbox(
    "Feature Descriptor",
    ["GLCM", "BiT"],
    help="GLCM: Gray Level Co-occurrence Matrix\nBiT: Binary Texture Descriptor"
)

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG, JPEG"
    )

    if uploaded_file:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("⚠️ Error loading image. Please try another image.")
        else:
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            if st.button("🔎 Find Similar Images"):
                with st.spinner('Processing image...'):
                    # Extract features
                    query_features = glcm(image) if descriptor == "GLCM" else bitdesc(image)
                    
                    # Search for similar images
                    results = retrieve_similar_images(signatures, query_features, distance_measure, num_images)
                    
                    if results:
                        with col2:
                            st.markdown("### Similar Images")
                            for i, (img_path, distance_val, label) in enumerate(results, 1):
                                try:
                                    img = cv2.imread(img_path)
                                    if img is not None:
                                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                        st.image(img, 
                                               caption=f'Match {i} | Distance: {distance_val:.4f} | Category: {label}',
                                               use_column_width=True)
                                    else:
                                        st.error(f"⚠️ Could not load image: {img_path}")
                                except Exception as e:
                                    st.error(f"⚠️ Error processing image {img_path}: {str(e)}")
                    else:
                        st.error("❌ No similar images found.")
    else:
        st.markdown("""
            <div class="upload-prompt">
                <h3>👆 Upload an image to start</h3>
                <p>Select an image file to find similar images in the database</p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit | Image Similarity Search Engine</p>
    </div>
""", unsafe_allow_html=True)