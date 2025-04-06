from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai
import requests
import time
from PIL import Image
import numpy as np
from io import BytesIO
import base64
import mimetypes

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-pro-vision")
chat_model = genai.GenerativeModel("gemini-pro")
chat = chat_model.start_chat(history=[])

# Function to detect disease from an uploaded image using Gemini Vision
def detect_disease_from_image(img):
    try:
        # Convert PIL Image to bytes
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        
        # Prepare the image part
        image_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_bytes).decode()
            }
        ]
        
        # Prepare prompt
        prompt_parts = [
            "Analyze this crop image and identify any visible diseases or problems.",
            "Focus specifically on agricultural diseases that affect plants.",
            "Provide only the name of the most likely disease in a single word or short phrase.",
            image_parts[0]
        ]
        
        # Get response from Gemini Vision
        response = model.generate_content(prompt_parts)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return "unknown disease"

# Function to get Gemini text response
def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

# Function to get weather data with improved error handling
def get_weather_data(city_name, api_key):
    if not api_key:
        st.error("OpenWeatherMap API key is missing. Please check your .env file.")
        return None

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city_name,
        "appid": api_key,
        "units": "metric"
    }
    
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            st.error("Invalid OpenWeatherMap API key. Please check your key and try again.")
        elif response.status_code == 404:
            st.error(f"City '{city_name}' not found. Please check the city name.")
        else:
            st.error(f"Failed to fetch weather data. Error code: {response.status_code}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error occurred while fetching weather data: {str(e)}")
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

if 'detected_disease' not in st.session_state:
    st.session_state['detected_disease'] = ""

if 'city_name' not in st.session_state:
    st.session_state['city_name'] = "Bangalore"

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

# Labels dictionary
labels = {
    'header': "AgriTech Titans",
    'input_label': "Input:",
    'city_label': "Enter your city name for weather data:",
    'generate_button': "Generate Report",
    'next_day_button': "Next Day Report",
    'show_history_button': "Show History",
    'translate_button': "Translate Report to Kannada",
    'weather_header': "Weather in",
    'report_title': "Day {} Report",
    'next_day_header': "The Next Day Report is",
    'translated_header': "Translated Report (Kannada)",
    'no_report_warning': "No report available to translate.",
    'no_response_warning': "No response was generated. Try simplifying your request.",
    'weather_error': "Failed to fetch weather data. Please check the city name or API key.",
    'cooldown_warning': "You can only generate a report once every 5 minutes. Please wait.",
    'api_limit_warning': "You have exceeded the maximum number of API calls. Please wait before making another request.",
    'view_kannada_button': "View Report in Kannada",
    'view_english_button': "View Report in English",
    'api_key_missing': "OpenWeatherMap API key is missing. Please check your .env file.",
    'invalid_api_key': "Invalid OpenWeatherMap API key. Please check your key and try again.",
    'city_not_found': "City not found. Please check the city name.",
}

# Function to apply translations
def apply_translations():
    global labels
    if st.session_state['language'] == 'kn':
        for key in labels:
            labels[key] = translate_text(labels[key], 'kn')

# Apply translations initially
apply_translations()

# Streamlit app configuration
st.set_page_config(page_title="AgriTech Titans", layout="wide")

# Sidebar for language selection and actions
with st.sidebar:
    st.markdown("### " + translate_text("Select Language", st.session_state['language']))
    if st.button(translate_text("English", st.session_state['language'])):
        if st.session_state['language'] != 'en':
            st.session_state['language'] = 'en'
            apply_translations()
            st.rerun()

    if st.button(translate_text("ಕನ್ನಡ (Kannada)", st.session_state['language'])):
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
            translation_prompt = f"Translate the following agricultural report into Kannada while keeping all technical terms accurate: {text_to_translate}"
            response = get_gemini_response(translation_prompt)

            st.session_state['daily_suggestions'][f'day{current_day}_kannada'] = ""
            for chunk in response:
                if hasattr(chunk, "text"):
                    st.session_state['daily_suggestions'][f'day{current_day}_kannada'] += chunk.text

            st.success(translate_text("Report translated to Kannada!", st.session_state['language']))
            st.session_state['view_kannada'] = True
        else:
            st.warning(labels['no_report_warning'])

    if st.session_state['show_history']:
        st.markdown("### " + translate_text("Chat History", st.session_state['language']))
        for role, text in st.session_state['chat_history']:
            st.markdown(f"**{role}:** {text}")

# Main content
st.header(labels['header'])

# Add an image upload feature in Streamlit
st.sidebar.header(translate_text("Upload Crop Image", st.session_state['language']))
uploaded_image = st.sidebar.file_uploader(translate_text("Upload an image of the diseased crop", st.session_state['language']), type=["jpg", "jpeg", "png"])

# Process the uploaded image and detect disease
if uploaded_image:
    st.image(uploaded_image, caption=translate_text("Uploaded Crop Image", st.session_state['language']), use_column_width=True)
    image = Image.open(uploaded_image)
    st.session_state['detected_disease'] = detect_disease_from_image(image)
    default_input_text = translate_text(
        f"Provide detailed agricultural solutions for {st.session_state['detected_disease']} including:\n1. Recommended organic and chemical treatments\n2. Application methods and schedules\n3. Weather-appropriate precautions\n4. Preventive measures\n5. Expected recovery timeline\n\nConsider current weather conditions in {st.session_state['city_name']} when making recommendations.",
        st.session_state['language']
    )
    st.text(translate_text(f"Detected Disease: {st.session_state['detected_disease']}", st.session_state['language']))
else:
    default_input_text = translate_text(
        "Provide detailed agricultural solutions for |disease name| including:\n1. Recommended organic and chemical treatments\n2. Application methods and schedules\n3. Weather-appropriate precautions\n4. Preventive measures\n5. Expected recovery timeline\n\nConsider current weather conditions when making recommendations.",
        st.session_state['language']
    )

# User input and city input
user_input = st.text_area(
    labels['input_label'],
    key="input",
    value=default_input_text,
    height=150
)

st.session_state['city_name'] = st.text_input(
    labels['city_label'],
    key="city_input",
    value=translate_text(st.session_state['city_name'], st.session_state['language'])
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

            weather_data = get_weather_data(st.session_state['city_name'], os.getenv("OPENWEATHERMAP_API_KEY"))
            if weather_data:
                weather_description = weather_data['weather'][0]['description']
                temperature = weather_data['main']['temp']
                humidity = weather_data['main']['humidity']
                wind_speed = weather_data['wind']['speed']
                weather_info = (
                    f"Current weather in {st.session_state['city_name']}:\n"
                    f"- Conditions: {weather_description}\n"
                    f"- Temperature: {temperature}°C\n"
                    f"- Humidity: {humidity}%\n"
                    f"- Wind speed: {wind_speed} m/s\n\n"
                    "Provide agricultural recommendations that account for these weather conditions."
                )
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
            st.success(translate_text("Report generated successfully!", st.session_state['language']))
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

                weather_data = get_weather_data(st.session_state['city_name'], os.getenv("OPENWEATHERMAP_API_KEY"))
                if weather_data:
                    weather_description = weather_data['weather'][0]['description']
                    temperature = weather_data['main']['temp']
                    humidity = weather_data['main']['humidity']
                    wind_speed = weather_data['wind']['speed']
                    weather_info = (
                        f"Current weather in {st.session_state['city_name']}:\n"
                        f"- Conditions: {weather_description}\n"
                        f"- Temperature: {temperature}°C\n"
                        f"- Humidity: {humidity}%\n"
                        f"- Wind speed: {wind_speed} m/s\n\n"
                        "Provide agricultural recommendations that account for these weather conditions."
                    )
                else:
                    weather_info = labels['weather_error']

                prompt = f"Based on the previous day's treatment plan, provide updated recommendations for day {st.session_state['current_day']} considering:\n{weather_info}"
                response = get_gemini_response(prompt)

                st.session_state['daily_suggestions'][f'day{st.session_state["current_day"]}'] = ""
                for chunk in response:
                    if hasattr(chunk, "text"):
                        st.session_state['daily_suggestions'][f'day{st.session_state["current_day"]}'] += chunk.text

                st.session_state['chat_history'].append(("User", f"Day {st.session_state['current_day']} update"))
                st.session_state['chat_history'].append(("AgriTech Titans", st.session_state['daily_suggestions'][f'day{st.session_state["current_day"]}']))
                st.success(translate_text("Next day report generated successfully!", st.session_state['language']))
            else:
                st.warning(labels['api_limit_warning'])
        else:
            st.warning(translate_text("Please generate the first report before generating the next day report.", st.session_state['language']))

# Display the generated report
if st.session_state['is_first_report_generated']:
    current_day = st.session_state['current_day']
    if f'day{current_day}' in st.session_state['daily_suggestions']:
        st.markdown(f"### {labels['report_title'].format(current_day)}")
        
        if st.session_state['view_kannada'] and f'day{current_day}_kannada' in st.session_state['daily_suggestions']:
            st.markdown(st.session_state['daily_suggestions'][f'day{current_day}_kannada'])
        else:
            report_text = st.session_state['daily_suggestions'][f'day{current_day}']
            
            # Format the report with better readability
            st.markdown("#### " + translate_text("Agricultural Recommendations", st.session_state['language']))
            st.markdown(report_text.replace("- ", "• ").replace("\n", "  \n"))
            
            # Add toggle for language view
            if f'day{current_day}_kannada' in st.session_state['daily_suggestions']:
                if st.session_state['view_kannada']:
                    if st.button(labels['view_english_button']):
                        st.session_state['view_kannada'] = False
                        st.rerun()
                else:
                    if st.button(labels['view_kannada_button']):
                        st.session_state['view_kannada'] = True
                        st.rerun()
    else:
        st.warning(labels['no_response_warning'])

# Display weather data
weather_api_key = os.getenv("OPENWEATHERMAP_API_KEY")
if not weather_api_key:
    st.error(labels['api_key_missing'])
else:
    weather_data = get_weather_data(st.session_state['city_name'], weather_api_key)
    if weather_data:
        st.markdown(f"### {labels['weather_header']} {st.session_state['city_name']}")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(translate_text("Temperature", st.session_state['language']), f"{weather_data['main']['temp']}°C")
        with col2:
            st.metric(translate_text("Humidity", st.session_state['language']), f"{weather_data['main']['humidity']}%")
        with col3:
            st.metric(translate_text("Conditions", st.session_state['language']), weather_data['weather'][0]['description'].title())
        with col4:
            st.metric(translate_text("Wind Speed", st.session_state['language']), f"{weather_data['wind']['speed']} m/s")
    elif weather_data is None:
        st.warning(labels['weather_error'])