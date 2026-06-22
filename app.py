import os
import tempfile

import streamlit as st
import whisper

from groq import Groq
from dotenv import load_dotenv


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


client = Groq(
    api_key=GROQ_API_KEY
)


@st.cache_resource
def load_whisper_model():
    """
    Loads Whisper only once.
    Prevents reloading every time Streamlit refreshes.
    """

    return whisper.load_model("base")


model = load_whisper_model()


st.set_page_config(
    page_title="AI Notes Summarizer",
    page_icon="🎙️",
    layout="wide"
)



st.title("🎙️ AI Notes Summarizer")

st.write(
    """
    Upload an audio recording.
    
    The application will:
    
    1. Convert speech into text
    2. Generate structured notes
    3. Display transcript and summary
    """
)



uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3", "m4a"]
)



if uploaded_file:

    st.audio(uploaded_file)

    if st.button("Generate Notes"):


        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp_file:

            temp_file.write(
                uploaded_file.read()
            )

            temp_audio_path = temp_file.name


        with st.spinner(
            "Transcribing audio..."
        ):

            transcription = model.transcribe(
                temp_audio_path
            )

            transcript = transcription["text"]


        prompt = f"""
You are an expert note-taking assistant.

Create concise and structured notes from the transcript.

Transcript:

{transcript}

Provide output in the following format:

Main Topics:
- item

Key Points:
- item

Action Items:
- item

Final Summary:
paragraph
"""


        with st.spinner(
            "Generating summary..."
        ):

            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3
            )

            summary = (
                response
                .choices[0]
                .message
                .content
            )


        st.success(
            "Notes generated successfully!"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Full Transcript"
            )

            st.text_area(
                "",
                transcript,
                height=400
            )

        with col2:

            st.subheader(
                "Generated Summary"
            )

            st.markdown(summary)


        st.download_button(
            label="Download Summary",
            data=summary,
            file_name="notes_summary.txt",
            mime="text/plain"
        )