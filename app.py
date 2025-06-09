import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from io import BytesIO
from PIL import Image

# --- Config ---
max_caption_length = 35  # Set this to your actual value
cnn_output_dim = 2048    # Update if you use a different CNN

# --- Load model & tokenizer ---

def load_model_and_tokenizer():
    model = tf.keras.models.load_model("caption_model.h5", compile=False)
    with open("image_features.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# --- Load feature extractor (InceptionV3) ---

def load_feature_extractor():
    base_model = InceptionV3(weights="imagenet")
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    return model

# --- Preprocess image and extract features ---
def preprocess_image_from_bytes(image_bytes):
    """Preprocess image from bytes"""
    img = Image.open(BytesIO(image_bytes))
    img = img.convert('RGB')  # Ensure RGB format
    img = img.resize((299, 299))  # Resize to InceptionV3 input size
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def preprocess_image(image_path):
    """Preprocess image from file path"""
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def extract_image_features(model, image_array):
    """Extract features from preprocessed image array"""
    features = model.predict(image_array, verbose=0)
    return features

# --- Caption generation using greedy search ---
def greedy_generator(image_features, tokenizer, model):
    in_text = 'start '
    for _ in range(max_caption_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_length).reshape((1, max_caption_length))
        prediction = model.predict([image_features, sequence], verbose=0)
        idx = np.argmax(prediction)
        word = tokenizer.index_word.get(idx)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    in_text = in_text.replace('start ', '').replace(' end', '')
    return in_text

# --- Streamlit UI ---
st.set_page_config(page_title="Image Captioning App", layout="centered")
st.title("🖼️ Image Caption Generator (Greedy Search)")
st.markdown("Upload an image and generate a caption using a pre-trained image captioning model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, use_column_width=True, caption="Uploaded Image")

    with st.spinner("Generating caption..."):
        try:
            # Load models
            model, tokenizer = load_model_and_tokenizer()
            feature_extractor = load_feature_extractor()

            # Process the uploaded image
            image_bytes = uploaded_file.read()
            preprocessed_image = preprocess_image_from_bytes(image_bytes)
            image_features = extract_image_features(feature_extractor, preprocessed_image)
            
            # Generate caption
            caption = greedy_generator(image_features, tokenizer, model)
            
            st.success("Generated Caption:")
            st.write(f"**{caption}**")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please make sure your model files ('caption_model.h5' and 'image_features.pkl') are in the correct location.")