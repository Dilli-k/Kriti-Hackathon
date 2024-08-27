import streamlit as st
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

# Import the necessary functions from utils.py
from utils import load_model, get_relevant_columns

# Load and display a banner image (Optional)
# banner = Image.open("banner_image.jpg")  # Replace with the path to your image
# st.image(banner, use_column_width=True)

# Load the model and tokenizer
model, tokenizer, device = load_model()

# Streamlit UI
st.title("📝 Text Input and Model Output")

# Sidebar for additional options
st.sidebar.header("Customization")
text_color = st.sidebar.color_picker("Pick a text color", "#000000")
background_color = st.sidebar.color_picker("Pick a background color", "#FFFFFF")
font_size = st.sidebar.slider("Select font size", 12, 36, 18)
st.sidebar.markdown("### About")
st.sidebar.info("This app allows you to input text and receive model-generated output. Customize the appearance using the options above!")

# Apply custom styles
st.markdown(
    f"""
    <style>
    .stTextInput textarea {{
        background-color: {background_color};
        color: {text_color};
        font-size: {font_size}px;
    }}
    .stTextInput div {{
        font-size: {font_size}px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Input text box
user_input = st.text_area("Enter your text here:")

if st.button("🔮 Generate Output"):
    # Generate model output based on user input
    output = get_relevant_columns(user_input, model, tokenizer, device)
    
    # Display the output with some styles
    st.subheader("Model Output:")
    st.markdown(
        f"""
        <div style="padding: 10px; border-radius: 5px; background-color: {background_color}; color: {text_color}; font-size: {font_size}px;">
        {output}
        </div>
        """,
        unsafe_allow_html=True,
    )

# Footer with a custom message
st.markdown("---")
st.markdown("✨ Developed by [Your name] | Powered by Streamlit")
