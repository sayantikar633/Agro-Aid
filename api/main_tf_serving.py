import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Set page configuration
st.set_page_config(page_title='Your Personal Agro-Aid !', layout='wide')

saved_model_path = "models/1"

# Check if the directory exists
if not os.path.exists(saved_model_path):
    raise FileNotFoundError(f"SavedModel directory '{saved_model_path}' does not exist.")

# Load the model
MODEL = tf.keras.models.load_model(saved_model_path)

CLASS_NAMES = ["Early Blight","Late Blight","Healthy"]

def read_file_as_image(data) ->np.ndarray:
    image = np.array(data)
    return image

# Set up sidebar
st.sidebar.title('Navigation')
page = st.sidebar.radio("Go to", ['Home', 'Disease Recognition', 'Treatment', 'News Updates', 'About'])

if page == 'Home':
    st.title('Welcome to your Personal Agro-Aid!!! 🌿🔍')
    st.write("""
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
    """)

elif page == 'Disease Recognition':
    st.title('Disease Recognition')
    uploaded_file = st.file_uploader("Upload your image", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        if st.button('Predict'):
            st.write("Predicting...")
            image = read_file_as_image(image)
            img_batch = np.expand_dims(image,0)
            predictions = MODEL.predict(img_batch)
            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            confidence = np.max(predictions[0])
            st.success(f"Class: {predicted_class}, Confidence: {confidence*100:.2f}%")
            if st.button('Predict Again'):
                uploaded_file = None
                st.write("Please upload the next image.")
        if st.button('Exit'):
            uploaded_file = None
            st.write("Exited. Please refresh the page for a new session.")

elif page == 'Treatment':
    st.title('Treatment')
    treatment_option = st.sidebar.radio("Choose a disease", ['Early Blight', 'Late Blight'])
    if treatment_option == 'Early Blight':
        st.write("""
        ### Early Blight Treatment
        1. **Fungicides:** Apply fungicides to protect plants, especially during periods of frequent rainfall.
        2. **Proper Spacing:** Space plants properly to improve air circulation and allow foliage to dry quickly.
        3. **Crop Rotation:** Practice crop rotation with non-host crops to reduce the disease inoculum in the soil.
        """)
        st.write("For more information, visit: https://shasyadhara.com/early-blight-of-potato-cause-symptoms-and-control/")
    elif treatment_option == 'Late Blight':
        st.write("""
        ### Late Blight Treatment
        1. **Fungicides:** Use fungicides as a preventive measure before the disease appears.
        2. **Destroy Infected Plants:** Remove and destroy all infected plants to prevent the spread of the disease.
        3. **Resistant Varieties:** Plant resistant varieties if they are available.
        """)
        st.write("For more information, visit: https://krishijagran.com/agripedia/late-blight-of-potato-complete-management-strategy-for-this-deadly-disease/")

elif page == 'News Updates':
    st.title('News Updates')
    st.write("Here are the latest news updates related to potato plants:")

    news_data = [
        {"title": "Potato News Today – No-nonsense, no-frills potato news stories from around the world", "link": "https://www.potatonewstoday.com/"},
        {"title": "HyFun Foods to invest Rs 850 crore for three potato processing plants in Gujarat", "link": "https://www.thehindubusinessline.com/companies/hyfun-foods-to-invest-rs-850-crore-for-three-potato-processing-plants-in-gujarat/article37185989.ece"},
        {"title": "\"Pomato\": Farmer's New Technique Of Growing Potatoes And Tomatoes On One Plant", "link": "https://www.ndtv.com/offbeat/pomato-farmers-new-technique-of-growing-potatoes-and-tomatoes-on-one-plant-see-viral-tweet-2617625"},
        {"title": "Big increase in United States potato crop expected", "link": "https://www.freshplaza.com/article/9366388/big-increase-in-united-states-potato-crop-expected/"},
    ]

    for news in news_data:
        st.markdown(f"[{news['title']}]({news['link']})")


elif page == 'About':
    st.title('About Us')
    st.write("""
    Learn more about the project, our team, and our goals on this page.
    """)
