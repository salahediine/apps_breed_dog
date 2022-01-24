import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import cv2 as cv
import streamlit as st
from tensorflow.keras.utils import img_to_array
import keras
from PIL import Image
import numpy as np
import keras


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model = load_model('xception_model.h5', custom_objects={"f1_m": f1_m})


# Displaying in-head
st.title("What is the breed of your dog ?")
st.image("banner.jpg")
st.text("Upload a picture of your dog to get its breed")


list_race = ['Chihuahua', 'French_bulldog', 'German_shepherd', 'German_short',
       'Labrador_retriever', 'Rottweiler', 'Yorkshire_terrier', 'beagle',
       'golden_retriever', 'toy_poodle']


def image_classifier(img):
 
  model = load_model('xception_model.h5', custom_objects={"f1_m": f1_m})

  img_array = img_to_array(img)
  img = cv.cvtColor(img_array,cv.COLOR_BGR2RGB)
  img = cv.resize(img, (255,255), interpolation=cv.INTER_LINEAR) # redimensionner
 #On convertit l'image transformée en array
  img_array = keras.preprocessing.image.img_to_array(img)

  img_array = img_array.reshape((-1, 255, 255, 3))
  img_array = tf.keras.applications.xception.preprocess_input(img_array) # appliquer le process xcpetion

# predictions
  prediction = model.predict(img_array).flatten()

  list_prob = [round(prediction[i],2) for i in range(10)]
  max_prob = max(list_prob)
  list_race_max = list_race[np.argmax(list_prob)]

  
  return list_race_max

# Displaying uploader
uploaded_file = st.file_uploader("Upload your image...", type="jpg")

# Loop ending with prediction
if uploaded_file is not None:
  img = Image.open(uploaded_file)
  st.image(img, caption='Uploaded image.', use_column_width=True)
  st.write("")
  st.write("Classifying...")
  label = image_classifier(img)
  st.write(label)
