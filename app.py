import streamlit as st
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model

# Load tokenizer and model

def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        return pickle.load(f)


def load_caption_model():
    return load_model('caption_model.h5', compile=False)


def load_feature_extractor():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    return Model(inputs=base_model.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output))

tokenizer = load_tokenizer()
caption_model = load_caption_model()
feature_extractor = load_feature_extractor()

# Set constants
max_caption_length = 38  # or your actual max length used during training
cnn_output_dim = 2048    # for InceptionV3's GlobalAveragePooling2D output

def preprocess_image(image):
    img = image.resize((299, 299)).convert('RGB')
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def extract_features(image, model):
    img = preprocess_image(image)
    return model.predict(img)[0]  # shape: (2048,)

# Your greedy_generator
def greedy_generator(image_features):
    in_text = 'start '
    for _ in range(max_caption_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_length).reshape((1, max_caption_length))
        prediction = caption_model.predict([image_features.reshape(1, cnn_output_dim), sequence], verbose=0)
        idx = np.argmax(prediction)
        word = tokenizer.index_word.get(idx)
        if word is None:
            break
        in_text += word + ' '
        if word == 'end':
            break
    in_text = in_text.replace('start ', '').replace(' end', '').strip()
    return in_text

# Streamlit UI
st.title("🖼️ Image Caption Generator (Greedy Search)")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner('Generating caption...'):
        features = extract_features(image, feature_extractor)
        caption = greedy_generator(features)

    st.success("Generated Caption:")
    st.markdown(f"**{caption}**")
