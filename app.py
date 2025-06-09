import streamlit as st
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
MAX_CAPTION_LENGTH = 38
CNN_OUTPUT_DIM = 2048

# Load tokenizer

def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        return pickle.load(f)

# Load caption generation model

def load_caption_model():
    return load_model('caption_model.h5', compile=False)

# Load image feature extractor (InceptionV3 with GAP layer)

def load_feature_extractor():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    gap_layer = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling')(base_model.output)
    return Model(inputs=base_model.input, outputs=gap_layer)

# Preprocess image
def preprocess_image_from_pil(pil_image):
    img = pil_image.resize((299, 299)).convert('RGB')
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Extract features from image
def extract_image_features_from_pil(model, pil_image):
    img = preprocess_image_from_pil(pil_image)
    features = model.predict(img, verbose=0)
    return features[0]

# Generate caption using greedy search
def greedy_generator(model, tokenizer, image_features):
    in_text = 'start'
    for _ in range(MAX_CAPTION_LENGTH):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_CAPTION_LENGTH)
        prediction = model.predict([np.expand_dims(image_features, axis=0), sequence], verbose=0)
        idx = np.argmax(prediction)
        word = tokenizer.index_word.get(idx)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text.replace('start ', '').replace(' end', '').strip()

# Load models
tokenizer = load_tokenizer()
caption_model = load_caption_model()
feature_extractor = load_feature_extractor()

# Streamlit UI
st.set_page_config(page_title="Caption Generator App", page_icon="📷")
st.title("🖼️ Image Caption Generator (Greedy Search)")
st.markdown("Upload an image and get a generated caption using a trained LSTM model with InceptionV3 features.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        features = extract_image_features_from_pil(feature_extractor, image)
        caption = greedy_generator(caption_model, tokenizer, features)

    st.success("Generated Caption:")
    st.markdown(f"**{caption}**")
