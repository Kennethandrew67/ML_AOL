import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import load_img, img_to_array
import numpy as np
import tensorflow as tf
import pickle

# Load model and tokenizer
@st.cache(allow_output_mutation=True)
def load_caption_model():
    model = tf.keras.models.load_model('caption_model.h5')
    with open('image_features.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# Preprocess uploaded image
def preprocess_image(image, cnn_output_dim, preprocess_input, feature_extractor):
    image = image.resize((299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = feature_extractor.predict(image)
    return features.reshape(1, cnn_output_dim)

# Generate caption
def generate_caption(model, tokenizer, features, max_len):
    in_text = 'start'
    for _ in range(max_len):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_len)
        prediction = model.predict([features, sequence], verbose=0)
        idx = np.argmax(prediction)
        word = tokenizer.index_word.get(idx)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text.replace('start ', '').replace(' end', '')

# Streamlit UI
st.title("🖼️ Image Caption Generator")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    with st.spinner("Generating caption..."):
        model, tokenizer = load_caption_model()
        max_len = 35  # your max_caption_length
        cnn_output_dim = 2048  # or 4096, depends on your model
        from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
        from tensorflow.keras.models import Model

        base_model = InceptionV3(weights='imagenet')
        feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

        image = load_img(uploaded_file, target_size=(299, 299))
        features = preprocess_image(image, cnn_output_dim, preprocess_input, feature_extractor)
        caption = generate_caption(model, tokenizer, features, max_len)

        st.subheader("📝 Generated Caption:")
        st.write(caption)
