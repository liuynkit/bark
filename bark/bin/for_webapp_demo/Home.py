import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# from IPython.display import Audio
import nltk  # we'll use this to split into sentences
import numpy as np
import streamlit as st
import scipy.io.wavfile
import string

from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE

@st.cache_resource()
def LoadingModels():
    preload_models()

def tts(script):
    sentences = nltk.sent_tokenize(script)
    SPEAKER = "v2/en_speaker_6"
    silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

    pieces = []
    for sentence in sentences:
        audio_array = generate_audio(sentence, history_prompt=SPEAKER)
        pieces += [audio_array, silence.copy()]

    wav = np.concatenate(pieces)
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))

    file_name = script.replace(" ", "_")[0:20]
    file_name = file_name.translate(
        str.maketrans('', '', string.punctuation.replace('_', ''))) + '.wav'
    path = f"temp/{file_name}"
    scipy.io.wavfile.write(path, SAMPLE_RATE, wav_norm.astype(np.int16))

    return path

script = """
Hey, have you heard about this new text-to-audio model called "Bark"? 
Apparently, it's the most realistic and natural-sounding text-to-audio model 
out there right now. People are saying it sounds just like a real person speaking. 
I think it uses advanced machine learning algorithms to analyze and understand the 
nuances of human speech, and then replicates those nuances in its own speech output. 
It's pretty impressive, and I bet it could be used for things like audiobooks or podcasts. 
In fact, I heard that some publishers are already starting to use Bark to create audiobooks. 
It would be like having your own personal voiceover artist. I really think Bark is going to 
be a game-changer in the world of text-to-audio technology.
""".replace("\n", " ").strip()

preload_models(text_use_small=True)

text = st.text_input("Enter text")

if st.button("Synthesis"): 
    if text=='':
        st.write("Use predefined script instead.")   
        text = script  

    path = tts(text)
    audio_file = open(path, "rb")
    audio_bytes = audio_file.read()
    st.markdown(f"## Your audio with model origin:")
    st.audio(audio_bytes, format="audio/wav", start_time=0)