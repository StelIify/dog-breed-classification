import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from matplotlib import image

labels = pd.read_csv("data/labels.csv")
unique_breeds = np.unique(labels.breed)

print(unique_breeds)

MODEL_PATH = "data/models/20210226-004258-full-data-mobilenetv2-Adam.h5"


@st.cache_data()
def load_model(model_path=MODEL_PATH):
    model = tf.keras.models.load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})
    return model


loaded_full_data_model = load_model()


def process_image(image, image_size=224):

    converted_image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    resized_image = tf.image.resize(converted_image, size=[image_size, image_size])

    return resized_image


def predict(user_image):
    st.write("Classifying...")
    data = tf.data.Dataset.from_tensors(tf.constant(user_image))
    data_batch = data.map(process_image).batch(32)
    predicted_probabilities = loaded_full_data_model.predict(data_batch)
    predicted_label = unique_breeds[np.argmax(predicted_probabilities)]
    st.write(f"{predicted_label} with {np.max(predicted_probabilities) * 100:.2f}% probability")
    st.image(user_image, caption='Uploaded Image.', use_column_width=True)


st.title("Dog Breed Classification")

uploaded_file = st.file_uploader("Choose an image", type="jpg")

if uploaded_file is not None:

    uploaded_image = image.imread(uploaded_file)
    predict(uploaded_image)
