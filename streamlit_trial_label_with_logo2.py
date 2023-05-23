import os
import cv2
import streamlit as st
import time
import datetime
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import pyttsx3


import base64
# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#     st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background-image: url(data:image/{"jpeg"};base64,{encoded_string.decode()});
#         background-size: cover
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
#     )
# add_bg_from_local('eye.jpg')    

def main():
    st.title("Place the tool in front of webcam")
    
    cap = cv2.VideoCapture(1)
    
    capture_button = st.button("Capture")
    stop_button = st.button("Stop")
    
    if capture_button:
        # Create a directory to save captured images
        os.makedirs("captured_images", exist_ok=True)
        
        count = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Unable to capture frame from webcam. Please try again.")
                break
                
            st.image(frame, channels="BGR")
            st.write("Image captured successfully!")
            
            # Generate a unique filename based on timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"captured_images/image_{timestamp}_{count}.jpg"
            
            # Save the captured image
            cv2.imwrite(filename, frame)
            
            count += 1
            time.sleep(1) # Wait for 1 second before capturing the next photo
            
            if stop_button or count > 1:
                break

        # Load the last captured image as the background image in the center
        if count > 1:
            last_captured_image = cv2.imread(filename)
            st.image(last_captured_image, caption='Background Image', use_column_width=True)
        
    cap.release()

if __name__ == "__main__":
    main()


model = load_model("F:/SEM 6/Metal Anamoly Detection/scrach_1/modelinn6.h5", compile=False)

engine = pyttsx3.init()

# You can always call this function where ever you want

def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    
    modified_logo = logo.resize((width, height))
    return modified_logo

my_logo = add_logo(logo_path="eye6.jpeg", width=1000, height=800)
st.sidebar.image(my_logo)



st.title("Garuda - Tool Anomaly Inspection")
st.set_option('deprecation.showfileUploaderEncoding', False)

# Prompt the user to enter the threshold values
threshold_1 = st.number_input("Enter the first threshold value:", value=0.1)
threshold_2 = st.number_input("Enter the second threshold value:", value=0.5)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the input image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make a prediction
    start_time = time.time()
    preds = model.predict(x)
    end_time = time.time()



# ...

    # Print the predicted class label

    if preds[0][0] > threshold_2:
        prediction = 'Rejected'
        speech = 'The sample is Rejected'
        st.markdown("<h1 style='text-align: center; color: red; font-weight: bold;'>QUALITY DECISION: Rejected &#10060;</h1>", unsafe_allow_html=True)
    elif preds[0][0] > threshold_1:
        prediction = 'Accepted'
        speech = 'The sample is Accepted'
        st.markdown("<h1 style='text-align: center; color: green; font-weight: bold;'>QUALITY DECISION: Accepted &#10004;</h1>", unsafe_allow_html=True)
    
    # if preds[0][0] > 0.5:
    #     prediction = 'Rejected'
    #     speech = 'The sample is Rejected'
    #     st.markdown("<h1 style='text-align: center; color: red; font-weight: bold;'>QUALITY DECISION: Rejected &#10060;</h1>", unsafe_allow_html=True)
    # else:
    #     prediction = 'Accepted'
    #     speech = 'The sample is Accepted'
    #     st.markdown("<h1 style='text-align: center; color: green; font-weight: bold;'>QUALITY DECISION: Accepted &#10004;</h1>", unsafe_allow_html=True)

    # Show the image
    time_taken = '{:.2f} ms'.format((end_time - start_time) * 1000)
    img = cv2.putText(np.array(img), time_taken, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
    st.image(img, caption='Input Image with Prediction Time', use_column_width=True)

    # Speak the prediction result
    engine.say(speech)
    engine.runAndWait()
    engine.endLoop(True)

if engine._inLoop:
    engine.endLoop()
    
#     engine = None

#     engine.startLoop(False)
# # engine.iterate() must be called inside externalLoop()

# if engine._inLoop:


    # engine.startLoop(False)
   
# # engine.iterate() must be called inside Server_Up.start()
# Server_Up = threading.Thread(target = Comm_Connection)
# Server_Up.start()
# engine.endLoop()
