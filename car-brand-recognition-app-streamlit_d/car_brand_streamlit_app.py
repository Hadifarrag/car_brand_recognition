import streamlit as st 
import h5py
import tensorflow as tf
import cv2 as cv
import numpy as np
from PIL import Image
class_labels = {
    'Chevrolet Avalanche Crew Cab 2012': 0,
    'Buick Enclave SUV 2012': 1,
    'BMW X6 SUV 2012': 2,
    'Hyundai Tucson SUV 2012': 3,
    'Chevrolet Corvette ZR1 2012': 4,
    'Chevrolet Camaro Convertible 2012': 5,
    'Mitsubishi Lancer Sedan 2012': 6,
    'Cadillac CTS-V Sedan 2012': 7,
    'HUMMER H3T Crew Cab 2010': 8,
    'Chevrolet Cobalt SS 2010': 9,
    'Chevrolet Tahoe Hybrid SUV 2012': 10,
    'BMW Z4 Convertible 2012': 11,
    'Buick Rainier SUV 2007': 12,
    'Chevrolet Sonic Sedan 2012': 13,
    'Hyundai Accent Sedan 2012': 14,
    'Bugatti Veyron 16.4 Coupe 2009': 15,
    'Buick Verano Sedan 2012': 16,
    'Audi R8 Coupe 2012': 17,
    'Cadillac SRX SUV 2012': 18,
    'Chevrolet Silverado 1500 Extended Cab 2012': 19,
    'Acura TL Sedan 2012': 20,
    'Jeep Liberty SUV 2012': 21,
    'Chevrolet Impala Sedan 2007': 22,
    'Cadillac Escalade EXT Crew Cab 2007': 23,
    'Bentley Continental GT Coupe 2012': 24,
    'Chevrolet HHR SS 2010': 25,
    'Chevrolet Express Van 2007': 26
}

def load_model_from_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        model = tf.keras.models.load_model(f)
    return model


def preprocess_image(image_path, target_size=(128, 128)):

    image = np.array(image_path)
    #image = cv.imread(image)
    # Resize image to target size
    image = cv.resize(image, target_size)
    # Normalize pixel values to [0, 1]
    image = image / 255.0
   # if image.mode != "RGB":
    #    image = image.convert("RGB")
    #Expand dimentions
    image = np.expand_dims(image, axis=0)
    return image



st.title("Car Classification App")
st.header("Upload an image of a car and get its Name.")
#uploading the image 
uploaded_file = st.file_uploader("upload the car image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded file as a PIL image
    image_input = Image.open(uploaded_file)
    
    # Display the image using Streamlit
    st.image(image_input, caption="Uploaded Image", use_column_width=True)


   # st.write(type(image_input))
    # Moved the button and prediction code inside the if block
if st.button("recognize this car "):
    image = preprocess_image(image_input)  # Now image is defined
    model = load_model_from_hdf5("D:\\graduation\\API\\model.hdf5")
    prediction = model.predict(image)
    predicted_class_label = np.argmax(prediction)
    car_name = list(class_labels.keys())[predicted_class_label]
    st.write("The predicted car is:", car_name)
