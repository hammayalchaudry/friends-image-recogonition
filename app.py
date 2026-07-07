# app.py
import streamlit as st
import numpy as np
import os
import pickle
import tempfile
import sys

# Try to import opencv with better error handling
try:
    import cv2
except ImportError:
    st.error("""
    ⚠️ OpenCV is not installed. Please check your requirements.txt.
    Make sure opencv-python-headless is in your requirements file.
    """)
    st.stop()

try:
    import face_recognition
except ImportError:
    st.error("""
    ⚠️ Face Recognition library is not installed.
    Make sure face-recognition is in your requirements.txt file.
    """)
    st.stop()

from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Friend Recognizer",
    page_icon="👥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
    <style>
        .main {
            padding: 2rem 1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.5rem 2rem;
            transition: all 0.3s;
        }
        .stButton button:hover {
            background-color: #45a049;
            transform: scale(1.02);
        }
        .upload-box {
            border: 2px dashed #4CAF50;
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            background-color: #f9f9f9;
        }
        .result-box {
            background-color: #f0f8ff;
            border-radius: 15px;
            padding: 1.5rem;
            margin-top: 1.5rem;
            border-left: 5px solid #4CAF50;
        }
        .friend-tag {
            display: inline-block;
            background-color: #4CAF50;
            color: white;
            padding: 0.2rem 1rem;
            border-radius: 20px;
            font-size: 1.2rem;
            font-weight: bold;
            margin-top: 0.5rem;
        }
        .unknown-tag {
            display: inline-block;
            background-color: #ff6b6b;
            color: white;
            padding: 0.2rem 1rem;
            border-radius: 20px;
            font-size: 1.2rem;
            font-weight: bold;
            margin-top: 0.5rem;
        }
        .st-emotion-cache-1v0mbdj {
            max-width: 800px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 2rem;
        }
        .footer {
            text-align: center;
            color: #95a5a6;
            margin-top: 3rem;
            font-size: 0.9rem;
            border-top: 1px solid #ecf0f1;
            padding-top: 1.5rem;
        }
        .error-box {
            background-color: #ffe6e6;
            border: 1px solid #ff6b6b;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Title Section
st.markdown("<h1>📸 Friend Face Recognizer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload a photo and let AI recognize your friends instantly! 🚀</p>", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.known_encodings = []
    st.session_state.known_names = []

def train_from_dataset():
    """Train the face recognition model from images in the imagebase folder"""
    KNOWN_FACES_DIR = "imagebase"
    known_encodings = []
    known_names = []
    
    if not os.path.exists(KNOWN_FACES_DIR):
        return [], []
    
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            name = os.path.splitext(filename)[0].capitalize()
            
            try:
                img = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if len(encs) > 0:
                    known_encodings.append(encs[0])
                    known_names.append(name)
            except Exception as e:
                st.warning(f"Could not process {filename}: {e}")
    
    return known_encodings, known_names

# Load the pre-trained model
@st.cache_resource
def load_model():
    """Load the pre-trained face recognition model from pickle file"""
    model_path = "face_recognition_model.pkl"
    
    # If model exists in current directory, load it
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
            if model_data['encodings'] and model_data['names']:
                return model_data['encodings'], model_data['names']
        except Exception as e:
            st.warning(f"Could not load model file: {e}")
    
    # Fallback: Try to train from images
    encodings, names = train_from_dataset()
    if encodings:
        return encodings, names
    
    return [], []

# Load the model
known_encodings, known_names = load_model()

if known_names:
    st.session_state.model_loaded = True
    st.session_state.known_encodings = known_encodings
    st.session_state.known_names = known_names
    
    # Show loaded friends
    st.markdown(f"""
    <div style="background-color: #e8f5e9; border-radius: 10px; padding: 1rem; margin-bottom: 1.5rem;">
        <p style="margin: 0; font-size: 0.95rem; color: #2e7d32;">
            ✅ <b>Model Ready!</b> Recognizes: {', '.join(known_names)}
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("""
    ⚠️ **No model loaded!**
    
    To fix this:
    1. Add a folder named `imagebase` to your repository
    2. Put photos of your friends in it (named like `amna.jpg`, `alishba.jpg`, etc.)
    3. Commit and push to GitHub
    4. Streamlit will auto-deploy the update
    """)

# Upload Section
st.markdown("### 📤 Upload a Photo")
st.markdown("Choose an image with faces to recognize your friends.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# Process uploaded image
if uploaded_file is not None and st.session_state.model_loaded:
    image_placeholder = st.empty()
    result_placeholder = st.empty()
    
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("❌ Could not read the image. Please try another file.")
        else:
            # Convert BGR to RGB for face_recognition
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            # Process each face
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(
                    st.session_state.known_encodings, 
                    face_encoding, 
                    tolerance=0.55
                )
                name = "Unknown"
                
                face_distances = face_recognition.face_distance(
                    st.session_state.known_encodings, 
                    face_encoding
                )
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = st.session_state.known_names[best_match_index]
                
                # Draw bounding box and label
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(image, (left, top), (right, bottom), color, 3)
                
                # Label background
                label_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                label_y = max(top - 10, label_size[1] + 10)
                cv2.rectangle(
                    image, 
                    (left, label_y - label_size[1] - 10), 
                    (left + label_size[0] + 10, label_y + 5),
                    color, 
                    -1
                )
                cv2.putText(
                    image,
                    name,
                    (left + 5, label_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )
            
            # Convert back to RGB for display
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Display the result
            with image_placeholder.container():
                st.image(display_image, caption="Recognition Result", use_container_width=True)
            
            # Show results summary
            with result_placeholder.container():
                st.markdown("---")
                st.markdown("### 🎯 Recognition Results")
                
                if len(face_locations) == 0:
                    st.warning("😕 No faces detected in the image. Try another photo with visible faces.")
                else:
                    # Count recognized faces
                    recognized_names = []
                    for encoding in face_encodings:
                        matches = face_recognition.compare_faces(
                            st.session_state.known_encodings, 
                            encoding, 
                            tolerance=0.55
                        )
                        name = "Unknown"
                        face_distances = face_recognition.face_distance(
                            st.session_state.known_encodings, 
                            encoding
                        )
                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = st.session_state.known_names[best_match_index]
                        recognized_names.append(name)
                    
                    known_count = sum(1 for n in recognized_names if n != "Unknown")
                    unknown_count = len(recognized_names) - known_count
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("👤 Total Faces", len(face_locations))
                    with col2:
                        st.metric("✅ Recognized", known_count)
                    with col3:
                        st.metric("❓ Unknown", unknown_count)
                    
                    if known_count > 0:
                        friends_found = list(set([n for n in recognized_names if n != "Unknown"]))
                        st.markdown(f"**Friends identified:** {', '.join(friends_found)}")
                        
                        if len(friends_found) == len(face_locations):
                            st.success("🎉 All faces recognized successfully!")
                        else:
                            st.info(f"👀 {unknown_count} face(s) could not be recognized.")
                    
    except Exception as e:
        st.error(f"❌ An error occurred while processing the image: {e}")

elif uploaded_file is not None and not st.session_state.model_loaded:
    st.error("⚠️ Model not loaded. Please add images to the 'imagebase' folder.")

# Empty state - show upload prompt
if uploaded_file is None:
    st.markdown("""
    <div class="upload-box">
        <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">📷 Drop your photo here</p>
        <p style="color: #7f8c8d; font-size: 0.9rem;">or click the button above to browse files</p>
        <p style="color: #95a5a6; font-size: 0.8rem; margin-top: 0.5rem;">Supported formats: JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.model_loaded:
        st.markdown(f"""
        <div style="background-color: #f8f9fa; border-radius: 10px; padding: 1rem; margin-top: 1.5rem;">
            <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">
                🧑‍🤝‍🧑 <b>Ready to recognize:</b> {', '.join(st.session_state.known_names)}
            </p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    Made with ❤️ using Face Recognition & Streamlit
</div>
""", unsafe_allow_html=True)
