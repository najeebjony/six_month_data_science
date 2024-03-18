import streamlit as st
from openai import OpenAI
import tempfile
import os

# Sidebar for API KEY Input 
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

# Main app title with logo
st.title("Speech-to-Text with Whisper")

st.write("This APP Created by Najeeb ullah")

audio_file = st.file_uploader("Upload a file", type=["mp3", "wav", "ogg", "flac"])
client = OpenAI(api_key=api_key)
if audio_file is not None and api_key:
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + audio_file.name.split(".")[-1]) as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_file.seek(0)
        audio_file_path = tmp_file.name

    try:
        with open(audio_file_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            transcription_text = response.text

            st.write("Transcription:", transcription_text)

    except Exception as e:
        st.error(f"An error occurred:{str(e)}")

    finally:
        os.remove(audio_file_path)
# Profile Information/Footer in Sidebar
# This markdown will serve as a footer in your sidebar
# It's placed after your main interactive elements
st.sidebar.markdown("""---""")  # Horizontal rule for visual separation
st.sidebar.markdown(
    """
    **Developer Profile**

    - **Name:** Najeeb ullah
    - **Facebbok:**[Facebook](https://www.facebook.com/Najeeb.shekih)
    - **GitHub:** [github](https://github.com/najeebjony)
    - **kaggel:**  [kaggel](https://www.kaggle.com/najeebjony)

    Made with ❤️ using Streamlit
    """,
    unsafe_allow_html=True
)
# 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
