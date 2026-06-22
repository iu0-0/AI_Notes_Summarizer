# AI Notes Summarizer

## Objective

Convert spoken audio into structured notes using AI.

## Features

- Upload audio files
- Speech-to-text using Whisper
- AI-generated notes using Llama 3 via Groq
- Display transcript
- Display summary
- Download summary

## Technologies Used

- Python
- Streamlit
- Whisper
- Groq API
- Llama 3

## Installation

pip install -r requirements.txt

## Run

streamlit run app.py

## Workflow

Audio Input
→ Whisper Speech-to-Text
→ Transcript
→ Groq Llama 3
→ Summary
→ Display Results