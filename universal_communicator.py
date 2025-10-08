import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoProcessor, AutoModelForSpeechSeq2Seq
from gtts import gTTS
import tempfile
import librosa
import torch
import io
import os

# Set page config
st.set_page_config(layout="wide", page_title="Universal Communicator")

# --- MODEL LOADING (ROBUST VERSION) ---
@st.cache_resource
def load_translator():
    """Loads the NLLB translation model robustly from a LOCAL folder."""
    st.write("Loading translation model from local files...")
    model_path = "nllb-model"
    if not os.path.exists(model_path):
        st.error(f"🔴 Translation model not found at '{model_path}'. Please run the manual download command.")
        st.stop()
    try:
        # Load tokenizer and model separately, ensuring offline mode is only for loading
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float16)
        
        # Create the pipeline from the pre-loaded components
        translator_pipeline = pipeline("translation", model=model, tokenizer=tokenizer)
        
        st.success("Translation model loaded successfully!")
        return translator_pipeline
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the local translation model: {e}")
        st.stop()


@st.cache_resource
def load_asr():
    """Loads the Whisper ASR model robustly from a LOCAL folder."""
    st.write("Loading speech recognition model from local files...")
    model_path = "whisper-model"
    if not os.path.exists(model_path):
        st.error(f"🔴 Speech model not found at '{model_path}'. Please run the manual download command.")
        st.stop()
    try:
        # Load processor and model separately for Whisper
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, local_files_only=True)
        
        # Create the pipeline from the pre-loaded components
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.float16
        )
        
        st.success("Speech recognition model loaded successfully!")
        return asr_pipeline
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the local speech model: {e}")
        st.stop()


# --- NLLB SUPPORTED LANGUAGES (Abridged for UI) ---
NLLB_LANGUAGES = {
    "English": "eng_Latn", "Spanish": "spa_Latn", "French": "fra_Latn",
    "German": "deu_Latn", "Hindi": "hin_Deva", "Chinese (Simplified)": "zho_Hans",
    "Arabic": "ara_Arab", "Russian": "rus_Cyrl", "Japanese": "jpn_Jpan",
    "Bengali": "ben_Beng"
}

# --- UI LAYOUT ---
st.title("🌍 Universal Communicator (Powered by AI Models)")
st.markdown("Translate text or your voice between hundreds of languages using state-of-the-art, open-source AI.")

# Load models
translator = load_translator()
asr_model = load_asr()

# Create two columns for Text and Voice translation
col1, col2 = st.columns(2)

# --- COLUMN 1: TEXT TRANSLATION ---
with col1:
    st.header("✍️ Text-to-Text Translation")
    text_input = st.text_area("Enter text to translate:", height=150, value="Hello, how are you today?")
    
    lang_keys = list(NLLB_LANGUAGES.keys())
    # Set default to English -> Bengali
    try:
        eng_index = lang_keys.index("English")
        ben_index = lang_keys.index("Bengali")
    except ValueError:
        eng_index, ben_index = 0, 4 # Fallback defaults

    src_lang_text = st.selectbox("Select source language:", lang_keys, index=eng_index, key="text_src")
    tgt_lang_text = st.selectbox("Select target language:", lang_keys, index=ben_index, key="text_tgt")

    if st.button("Translate Text"):
        if text_input.strip():
            with st.spinner("Translating..."):
                src_code = NLLB_LANGUAGES[src_lang_text]
                tgt_code = NLLB_LANGUAGES[tgt_lang_text]
                
                # The pipeline call now works correctly
                translated_text = translator(text_input, src_lang=src_code, tgt_lang=tgt_code)[0]['translation_text']
                
                st.subheader("Translated Text:")
                st.markdown(f"> {translated_text}")

                try:
                    # gTTS uses 2-letter codes (e.g., 'en' from 'eng_Latn')
                    lang_code_for_tts = tgt_code.split('_')[0]
                    tts = gTTS(text=translated_text, lang=lang_code_for_tts)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                        tts.save(fp.name)
                        st.audio(fp.name, format="audio/mp3")
                except Exception as e:
                    st.info(f"Could not generate audio for the selected language. Error: {e}")
        else:
            st.warning("Please enter some text to translate.")


# --- COLUMN 2: VOICE TRANSLATION ---
with col2:
    st.header("🎤 Voice-to-Text Translation")
    
    uploaded_audio = st.file_uploader("Upload an audio file (MP3, WAV)...", type=["mp3", "wav", "m4a"])
    
    audio_source = uploaded_audio

    if audio_source:
        st.audio(audio_source)
        
        st.write("**Translate this audio into:**")
        tgt_lang_voice_key = st.selectbox("Target Language:", lang_keys, index=ben_index, key="voice_tgt")

        if st.button("Translate Voice"):
            with st.spinner("Transcribing and translating audio... This may take a moment."):
                audio_bytes = audio_source.read()
                
                # Whisper can process raw bytes directly
                transcription_result = asr_model(audio_bytes)
                transcription = transcription_result["text"]
                st.subheader("Transcription (Detected Language):")
                st.markdown(f"> {transcription}")
                
                # For NLLB, we must specify the source language. We will default to English.
                # A more advanced app would use a language detection model first.
                src_code = NLLB_LANGUAGES["English"] 
                tgt_code = NLLB_LANGUAGES[tgt_lang_voice_key]

                translated_text = translator(transcription, src_lang=src_code, tgt_lang=tgt_code)[0]['translation_text']
                
                st.subheader("Final Translation:")
                st.markdown(f"> {translated_text}")

                try:
                    lang_code_for_tts = tgt_code.split('_')[0]
                    tts = gTTS(text=translated_text, lang=lang_code_for_tts)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                        tts.save(fp.name)
                        st.audio(fp.name, format="audio/mp3")
                except Exception as e:
                    st.info(f"Could not generate audio for the target language. Error: {e}")

