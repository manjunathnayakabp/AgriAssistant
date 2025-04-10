from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai
import requests
import time
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

# Load a pre-trained model for disease detection (e.g., MobileNetV2)
disease_model = MobileNetV2(weights="imagenet")

# Function to detect disease from an uploaded image
def detect_disease_from_image(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = disease_model.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
    predicted_disease = decoded_predictions[0][1]
    return predicted_disease

# Function to get Gemini response
def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

# Function to get weather data
def get_weather_data(city_name, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city_name,
        "appid": api_key,
        "units": "metric"
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Initialize session state variables
if 'language' not in st.session_state:
    st.session_state['language'] = 'en'

if 'show_history' not in st.session_state:
    st.session_state['show_history'] = False

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'daily_suggestions' not in st.session_state:
    st.session_state['daily_suggestions'] = {}

if 'current_day' not in st.session_state:
    st.session_state['current_day'] = 1

if 'last_report_time' not in st.session_state:
    st.session_state['last_report_time'] = 0

if 'api_call_count' not in st.session_state:
    st.session_state['api_call_count'] = 0

if 'last_api_call_time' not in st.session_state:
    st.session_state['last_api_call_time'] = 0

if 'is_generating_report' not in st.session_state:
    st.session_state['is_generating_report'] = False

if 'is_first_report_generated' not in st.session_state:
    st.session_state['is_first_report_generated'] = False

if 'view_kannada' not in st.session_state:
    st.session_state['view_kannada'] = False

# Cooldown period in seconds
COOLDOWN_PERIOD = 300

# API rate-limiting settings
MAX_API_CALLS = 5
API_COOLDOWN_PERIOD = 300

# Function to check if API calls are allowed
def is_api_call_allowed():
    current_time = time.time()
    last_api_call_time = st.session_state['last_api_call_time']
    api_call_count = st.session_state['api_call_count']

    if current_time - last_api_call_time > API_COOLDOWN_PERIOD:
        st.session_state['api_call_count'] = 0
        st.session_state['last_api_call_time'] = current_time
        return True

    if api_call_count >= MAX_API_CALLS:
        return False

    return True

# Function to increment API call count
def increment_api_call_count():
    st.session_state['api_call_count'] += 1
    st.session_state['last_api_call_time'] = time.time()

# Function to translate text
def translate_text(text, target_language):
    if st.session_state['language'] == target_language:
        return text
    translation_prompt = f"Translate the following text into {target_language}: {text}"
    response = get_gemini_response(translation_prompt)
    translated_text = ""
    for chunk in response:
        if hasattr(chunk, "text"):
            translated_text += chunk.text
    return translated_text

# Function to apply translations
def apply_translations():
    global labels
    labels = {
        'header': translate_text("AgriTech Titans", st.session_state['language']),
        'input_label': translate_text("Input:", st.session_state['language']),
        'city_label': translate_text("Enter your city name for weather data:", st.session_state['language']),
        'generate_button': translate_text("Generate Report", st.session_state['language']),
        'next_day_button': translate_text("Next Day Report", st.session_state['language']),
        'show_history_button': translate_text("Show History", st.session_state['language']),
        'translate_button': translate_text("Translate Report to Kannada", st.session_state['language']),
        'weather_header': translate_text("Weather in", st.session_state['language']),
        'report_title': translate_text("Day {} Report", st.session_state['language']),
        'next_day_header': translate_text("The Next Day Report is", st.session_state['language']),
        'translated_header': translate_text("Translated Report (Kannada)", st.session_state['language']),
        'no_report_warning': translate_text("No report available to translate.", st.session_state['language']),
        'no_response_warning': translate_text("No response was generated. Try simplifying your request.", st.session_state['language']),
        'weather_error': translate_text("Failed to fetch weather data. Please check the city name or API key.", st.session_state['language']),
        'cooldown_warning': translate_text("You can only generate a report once every 5 minutes. Please wait.", st.session_state['language']),
        'api_limit_warning': translate_text("You have exceeded the maximum number of API calls. Please wait before making another request.", st.session_state['language']),
        'view_kannada_button': translate_text("View Report in Kannada", st.session_state['language']),
        'view_english_button': translate_text("View Report in English", st.session_state['language']),
    }

# Apply translations initially
apply_translations()

# Streamlit app configuration
st.set_page_config(page_title="AgriTech Titans", layout="wide")

# Sidebar for language selection and actions
with st.sidebar:
    st.markdown("### Select Language")
    if st.button("English"):
        if st.session_state['language'] != 'en':
            st.session_state['language'] = 'en'
            apply_translations()
            st.rerun()

    if st.button("ಕನ್ನಡ (Kannada)"):
        if st.session_state['language'] != 'kn':
            st.session_state['language'] = 'kn'
            apply_translations()
            st.rerun()

    if st.button(labels['show_history_button']):
        st.session_state['show_history'] = not st.session_state['show_history']

    if st.button(labels['translate_button']):
        current_day = st.session_state['current_day']
        if f'day{current_day}' in st.session_state['daily_suggestions']:
            text_to_translate = st.session_state['daily_suggestions'][f'day{current_day}']
            translation_prompt = f"Translate the following text into Kannada language only .If the report is in english translate to english: {text_to_translate}"
            response = get_gemini_response(translation_prompt)

            st.session_state['daily_suggestions'][f'day{current_day}_kannada'] = ""
            for chunk in response:
                if hasattr(chunk, "text"):
                    st.session_state['daily_suggestions'][f'day{current_day}_kannada'] += chunk.text

            st.success("Report translated to Kannada!")
            st.session_state['view_kannada'] = True
        else:
            st.warning(labels['no_report_warning'])

    if st.session_state['show_history']:
        st.markdown("### Chat History")
        for role, text in st.session_state['chat_history']:
            st.markdown(f"{role}:** {text}", unsafe_allow_html=True)

# Main content
st.header(labels['header'])

# Add an image upload feature in Streamlit
st.sidebar.header("Upload Crop Image")
uploaded_image = st.sidebar.file_uploader("Upload an image of the diseased crop", type=["jpg", "jpeg", "png"])

# Process the uploaded image and detect disease
if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Crop Image", use_column_width=True)
    image = Image.open(uploaded_image)
    detected_disease = detect_disease_from_image(image)
    default_input_text = f"Provide agricultural solutions for the disease |{detected_disease}| and suggest fertilizers. Provide detailed usage instructions for the first 5 days, considering real-time weather conditions in your location. Focus only on agriculture and farming-related advice."
    st.text("Detected Disease: " + detected_disease)
else:
    default_input_text = "Provide agricultural solutions for the disease |disease name| and suggest fertilizers. Provide detailed usage instructions for the first 5 days, considering real-time weather conditions in your location. Focus only on agriculture and farming-related advice."

# User input and city input
user_input = st.text_input(
    labels['input_label'],
    key="input",
    value=translate_text(default_input_text, st.session_state['language'])
)

city_name = st.text_input(
    labels['city_label'],
    key="city_input",
    value=translate_text("Bangalore", st.session_state['language'])
)

# Buttons for generating report and next day report
col1, col2 = st.columns(2)

with col1:
    if st.button(labels['generate_button']):
        if is_api_call_allowed():
            increment_api_call_count()
            st.session_state['is_generating_report'] = True
            st.session_state['is_first_report_generated'] = True
            st.session_state['current_day'] = 1
            st.session_state['last_report_time'] = time.time()
            st.session_state['view_kannada'] = False

            weather_data = get_weather_data(city_name, os.getenv("OPENWEATHERMAP_API_KEY"))
            if weather_data:
                weather_description = weather_data['weather'][0]['description']
                temperature = weather_data['main']['temp']
                humidity = weather_data['main']['humidity']
                weather_info = f"Weather in {city_name}: {weather_description}, Temperature: {temperature}°C, Humidity: {humidity}%"
            else:
                weather_info = labels['weather_error']

            prompt = f"{user_input}\n\n{weather_info}"
            response = get_gemini_response(prompt)

            st.session_state['daily_suggestions'][f'day{st.session_state["current_day"]}'] = ""
            for chunk in response:
                if hasattr(chunk, "text"):
                    st.session_state['daily_suggestions'][f'day{st.session_state["current_day"]}'] += chunk.text

            st.session_state['chat_history'].append(("User", user_input))
            st.session_state['chat_history'].append(("AgriTech Titans", st.session_state['daily_suggestions'][f'day{st.session_state["current_day"]}']))
            st.success("Report generated successfully!")
        else:
            st.warning(labels['api_limit_warning'])

with col2:
    if st.button(labels['next_day_button']):
        if st.session_state['is_first_report_generated']:
            if is_api_call_allowed():
                increment_api_call_count()
                st.session_state['current_day'] += 1
                st.session_state['last_report_time'] = time.time()
                st.session_state['view_kannada'] = False

                weather_data = get_weather_data(city_name, os.getenv("OPENWEATHERMAP_API_KEY"))
                if weather_data:
                    weather_description = weather_data['weather'][0]['description']
                    temperature = weather_data['main']['temp']
                    humidity = weather_data['main']['humidity']
                    weather_info = f"Weather in {city_name}: {weather_description}, Temperature: {temperature}°C, Humidity: {humidity}%"
                else:
                    weather_info = labels['weather_error']

                prompt = f"{user_input}\n\n{weather_info}"
                response = get_gemini_response(prompt)

                st.session_state['daily_suggestions'][f'day{st.session_state["current_day"]}'] = ""
                for chunk in response:
                    if hasattr(chunk, "text"):
                        st.session_state['daily_suggestions'][f'day{st.session_state["current_day"]}'] += chunk.text

                st.session_state['chat_history'].append(("User", user_input))
                st.session_state['chat_history'].append(("AgriTech Titans", st.session_state['daily_suggestions'][f'day{st.session_state["current_day"]}']))
                st.success("Next day report generated successfully!")
            else:
                st.warning(labels['api_limit_warning'])
        else:
            st.warning("Please generate the first report before generating the next day report.")

# Display the generated report
if st.session_state['is_first_report_generated']:
    current_day = st.session_state['current_day']
    if f'day{current_day}' in st.session_state['daily_suggestions']:
        st.markdown(f"### {labels['report_title'].format(current_day)}")
        if st.session_state['view_kannada'] and f'day{current_day}_kannada' in st.session_state['daily_suggestions']:
            st.markdown(st.session_state['daily_suggestions'][f'day{current_day}_kannada'])
        else:
            st.markdown(st.session_state['daily_suggestions'][f'day{current_day}'])
    else:
        st.warning(labels['no_response_warning'])

# Display weather data
weather_data = get_weather_data(city_name, os.getenv("OPENWEATHERMAP_API_KEY"))
if weather_data:
    st.markdown(f"### {labels['weather_header']} {city_name}")
    st.write(f"Weather: {weather_data['weather'][0]['description']}")
    st.write(f"Temperature: {weather_data['main']['temp']}°C")
    st.write(f"Humidity: {weather_data['main']['humidity']}%")
else:
    st.warning(labels['weather_error'])