import streamlit as st
from openai import OpenAI
import tempfile
import os

# function to convert text to speech, modified to explicitly use an api key
def text_to_speech(api_key, text):
    """Converts text to speech using the OpenAI API tts-1 model and saves the output as a .mp3 file.
    Explicitly using an api key for authentication"""
    client = OpenAI(api_key=api_key)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        speech_file_path = tmpfile.name
        response = client.audio.speech.create(
            model=model,
            voice=voice,  # Replace with the desired voice
            input=text
        )

        response.stream_to_file(speech_file_path)

        return speech_file_path

st.title("Text-to-Speech converter")
st.image("http://www.piecex.com/product_image/20190625044028-00000544-image2.png")
st.markdown("""
This app convert text to speech using OpenAI's tts-1 model.
Please enter your OpenAI API key **Do not share your API key with others.**
""")
api_key = st.text_input("Enter your OpenAI key", type="password")


model = st.selectbox("Select a model", ["tts-1", "tts-1-hd"])

voice = st.selectbox("Select a voice", ["alloy","echo","feble","onyx","nova","shimmer"])

user_input = st.text_area("Enter your text here:", "Hello, welcome to our text to speech app.")

if st.button("Convert"):
    if not api_key:
        st.error("OpenAI API key is required to convert text to speech.")
    else:
        try:
            audio_file = text_to_speech(api_key, user_input)
            st.audio(open(audio_file, "rb"), format="audio/mp3")
            os.remove(audio_file)
        except Exception as e:
            st.error(f"An error occurred while converting text to speech: {str(e)}")

# add a download button to dowmload the final audio


st.sidebar.markdown("""---""")
st.sidebar.title("About")
st.sidebar.info("This app is created by [Najeeb](https://github.com/Najeebullah).")

# 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


