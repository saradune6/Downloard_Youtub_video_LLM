import streamlit as st
import os
import time
import requests
import yt_dlp
from pathlib import Path
from langchain_community.document_loaders import TextLoader  # Updated import
from langchain.indexes import VectorstoreIndexCreator
from langchain_groq import ChatGroq  
import re
from dotenv import load_dotenv
import subprocess

# Load API keys from the .env file
load_dotenv()
api_token = os.getenv('ASSEMBLY_AI_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

base_url = "https://api.assemblyai.com/v2"
headers = {
    "authorization": api_token,
    "content-type": "application/json"
}

# Function to download video from YouTube using yt_dlp
def save_video(url):
    try:
        os.makedirs('downloads', exist_ok=True)
        
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            file_id = info_dict.get("id")
            file_ext = "mkv"
            video_path = f'downloads/{file_id}.{file_ext}'
            
            if os.path.exists(video_path):
                st.info("Video already exists. Skipping download.")
                return video_path
            
            ydl_opts = {
                'format': 'bestvideo+bestaudio/best',
                'outtmpl': video_path,
                'ffmpeg_location': '/opt/homebrew/bin/ffmpeg',
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        
        return video_path
    except Exception as e:
        st.error(f"An error occurred while downloading the video: {e}")
        return None

# Function to extract audio from the downloaded video
def extract_audio(video_path):
    try:
        audio_path = video_path.rsplit('.', 1)[0] + ".mp3"
        if os.path.exists(audio_path):
            st.info("Audio already extracted. Skipping extraction.")
            return audio_path
        
        command = [
            'ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path
        ]
        subprocess.run(command, check=True)
        return audio_path
    except Exception as e:
        st.error(f"An error occurred while extracting audio: {e}")
        return None

# Function to transcribe audio using AssemblyAI
def assemblyai_stt(audio_path):
    try:
        with open(audio_path, "rb") as f:
            upload_response = requests.post(base_url + "/upload", headers=headers, data=f)
            if upload_response.status_code != 200:
                st.error(f"Upload failed: {upload_response.text}")
                return None

            upload_url = upload_response.json().get("upload_url")
            data = {"audio_url": upload_url}
            transcription_response = requests.post(base_url + "/transcript", json=data, headers=headers)
            if transcription_response.status_code != 200:
                st.error(f"Transcription request failed: {transcription_response.text}")
                return None

            transcript_id = transcription_response.json().get('id')
            polling_endpoint = f"{base_url}/transcript/{transcript_id}"
            while True:
                transcription_result = requests.get(polling_endpoint, headers=headers).json()
                if transcription_result.get('status') == 'completed':
                    return transcription_result.get('text')
                elif transcription_result.get('status') == 'error':
                    st.error(f"Transcription failed: {transcription_result['error']}")
                    return None
                time.sleep(3)
    except Exception as e:
        st.error(f"An error occurred during transcription: {e}")
        return None

# Function to validate YouTube URL format
def is_valid_url(url):
    youtube_pattern = r"^(https?:\/\/)?(www\.youtube\.com|youtu\.be)\/.+$"
    return re.match(youtube_pattern, url) is not None

# Streamlit UI Setup
st.title("Download and Summarize YouTube Video using LLM")

input_source = st.text_input("Enter the YouTube video URL")

if input_source:
    if is_valid_url(input_source):
        col1, col2 = st.columns(2)

        with col1:
            st.info("Your uploaded video")
            st.video(input_source)

            video_filename = save_video(input_source)
            if video_filename:
                audio_filename = extract_audio(video_filename)
                if audio_filename:
                    transcription = assemblyai_stt(audio_filename)
                    if transcription:
                        st.info(f"Transcription: {transcription[:500]} ...")  # Preview transcription

        with col2:
            st.info("Summary Below")
            if transcription:
                llm = ChatGroq(api_key=groq_api_key, model_name="mixtral-8x7b-32768")
                summary_prompt = [{"role": "user", "content": f"Summarize the following text: {transcription}"}]
                summary_response = llm.invoke(summary_prompt)

                # Ensure summary is correctly extracted
                if isinstance(summary_response, str):
                    summary_text = summary_response.strip()
                elif isinstance(summary_response, dict):
                    summary_text = summary_response.get("content", "").strip()
                elif hasattr(summary_response, "content"):
                    summary_text = summary_response.content.strip()
                elif hasattr(summary_response, "text"):
                    summary_text = summary_response.text.strip()
                else:
                    summary_text = "Summary extraction failed."

                st.success(summary_text)

    else:
        st.error("Please enter a valid YouTube URL.")
