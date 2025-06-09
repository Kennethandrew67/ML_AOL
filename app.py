import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model

# --- Config ---
max_caption_length = 35  # Set this to your actual value
cnn_output_dim = 2048    # Update if you use a different CNN

# --- Load model & tokenizer ---
@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("caption_model.h5", compile=False)
    with open("image_features.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# --- Load feature extractor (InceptionV3) ---
@st.cache(allow_output_mutation=True)
def load_feature_extractor():
    base_model = InceptionV3(weights="imagenet")
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    return model

# --- Preprocess image and extract features ---
def extract_features(image, feature_extractor):
    image = image.resize((299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = feature_extractor.predict(image)
    return features.reshape(1, cnn_output_dim)

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

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, use_column_width=True, caption="Uploaded Image")

    with st.spinner("Generating caption..."):
        model, tokenizer = load_model_and_tokenizer()
        feature_extractor = load_feature_extractor()

        image = load_img(uploaded_file)
        image_features = extract_features(image, feature_extractor)

        caption = greedy_generator(image_features, tokenizer, model)

        st.subheader("📝 Generated Caption:")
        st.write(caption)
