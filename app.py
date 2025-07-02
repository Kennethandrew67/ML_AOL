import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import Input


# Load InceptionV3 base model without top
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Add GlobalAveragePooling2D layer on top
x = GlobalAveragePooling2D()(base_model.output)
inception_model = Model(inputs=base_model.input, outputs=x)

cnn_output_dim = 2048


model = tf.keras.models.load_model('caption_model.h5', compile=False)

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

max_caption_length = 34

# Set custom web page title
st.set_page_config(page_title="Caption Generator App", page_icon="📷")

# Streamlit app
st.title("Image Caption Generator")
st.markdown("Upload an image, and this app will generate a caption.")

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Helper functions
def get_word_from_index(index, tokenizer):
    return next((word for word, idx in tokenizer.word_index.items() if idx == index), None)

def greedy_generator(image_features):
    in_text = 'start '
    for _ in range(max_caption_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_length).reshape((1, max_caption_length))
        prediction = model.predict([image_features.reshape(1, cnn_output_dim), sequence], verbose=0)
        idx = np.argmax(prediction)
        word = tokenizer.index_word.get(idx)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text.replace('start ', '').replace(' end', '')


# Process uploaded image
if uploaded_image is not None:
    st.subheader("Uploaded Image")
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Generated Caption")
    with st.spinner("Generating caption..."):
        # Preprocess uploaded image
        image = Image.open(uploaded_image).resize((299, 299)).convert('RGB')
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # Extract image features
        image_features = inception_model.predict(image, verbose=0)

        # Generate caption
        generated_caption = greedy_generator(image_features)
        st.markdown(f"**{generated_caption}**")
