import streamlit as st
from PIL import Image
import numpy as np
import pickle

# Load the saved model
with open('saved_steps.pkl', 'rb') as file:
   model = pickle.load(file)

st.title("Potato Disease Detection App")

# Drag and drop image upload
uploaded_file = st.file_uploader("Choose an image of potato leaf...", type=["jpg", "png"])
class_names = ['Potato-Early_blight', 'Potato-Late blight', 'Healthy Potato']
if uploaded_file is not None:
   # Open and display the image
   image = Image.open(uploaded_file)
   st.image(image, caption='Uploaded Image', use_column_width=True)
   st.write("")

   # Preprocess the image to fit the model input requirements
   image = image.resize((256, 256))
   image = np.array(image)
   image = image / 255.0  # Normalize to 0-1 range
   image = np.expand_dims(image, axis=0)  # Add batch dimension

   # batch_prediction = model.predict(image)
   # Predict the class
   predictions = model.predict(image)
   print("predicted label:",class_names[np.argmax(predictions[0])])
   # class_index = np.argmax(predictions)
   # print(predictions.shape)
   # print(predictions[0][0])
   # print(predictions[0][1])
   # print(predictions[0][2])

   st.subheader(f"Predicted class: {class_names[np.argmax(predictions[0])]}")
