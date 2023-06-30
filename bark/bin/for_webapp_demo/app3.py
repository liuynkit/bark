import streamlit as st
import os
import time
import string
from glob import glob
import torch
import matplotlib.pyplot as plt
from matplotlib import ticker
# from googletrans import Translator
# from time import sleep
# from stqdm import stqdm
# try:
#     os.mkdir("temp")
# except:
#     pass

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =""
PROJECT =""
REGION =""

st.title("Multi-lingual Multi-speaker TTS")

@st.cache_resource()
def LoadingModels():
    import sys
    from os.path import abspath, dirname
    sys.path.append(abspath(dirname(__file__)+'/../../'))
    from TTS.utils.synthesizer import Synthesizer

    # output_path = '/project/tts/students/yining_ws/multi_lng/TTS/outputs_multi_lingual/phoneme_version'
    # continue_path = max(glob(os.path.join(output_path, "multi_lingual_original_but_with_short-March-20-2023_07+47PM*/")), key=os.path.getmtime)
    output_path = '/export/data1/yliu/checkpoints'
    continue_path = max(glob(os.path.join(output_path, "multi_lingual_baseline_with_blank_with_lang_with_ar-April-28-2023_11+42AM*/")), key=os.path.getmtime)
    # continue_path = max(glob(os.path.join(output_path, "multi_lingual_ablation_version0-February-27-2023_12+15PM*/")), key=os.path.getmtime)
    config_path = os.path.join(continue_path, "config.json")
    model_path = os.path.join(continue_path, "best_model.pth")

    synthesizer_origin = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        tts_speakers_file=None,
        tts_languages_file=None,
        vocoder_checkpoint=None,
        vocoder_config=None,
        encoder_checkpoint=None,
        encoder_config=None,
        use_cuda=True,
    )
    return synthesizer_origin


text = st.text_input("Enter text")
in_lang = st.selectbox(
    "Select your language",
    ("English (American)", "English (British)", "Spanish", "French", "German", "Chinese", "Japanese", "Arabic"),
)
if in_lang == "English (American)":
    input_language = "en_US"
    options = ["Native American Female", "Native British Male"] 
if in_lang == "English (British)":
    input_language = "en_GB"
    options = ["Native British Male", "Native American Female"]
elif in_lang == "Spanish":
    input_language = "es_ES"
    options = ["Native Spanish Male"]
elif in_lang == "French":
    input_language = "fr_FR"
    options = ["Native French Female", "Native French Male"]
elif in_lang == "German":
    input_language = "de_DE"
    options = ["Native German Female", "Native German Male"]
elif in_lang == "Chinese":
    input_language = "zh_CN"
    options = ["Native Chinese Female"]
elif in_lang == "Japanese":
    input_language = "ja_JP"
    options = ["Native Japanese Female", "Native Japanese Male"]
elif in_lang == "Arabic":
    input_language = "ar_AR"
    options = ["Native Arabic Male"]

out_speaker = st.selectbox(
    "Select your speaker",
    options=options,
)
if out_speaker == "Native American Female":
    speaker_idx = "ljspeech"
elif out_speaker == "Native British Male":
    speaker_idx = "hifi_phil_benson"
elif out_speaker == "Native Spanish Male":
    speaker_idx = "css10_spanish"
elif out_speaker == "Native French Female":
    speaker_idx = "siwis_french"
elif out_speaker == "Native French Male":
    speaker_idx = "css10_french"
elif out_speaker == "Native Chinese Female":
    speaker_idx = "baker"
elif out_speaker == "Native German Female":
    speaker_idx = "Hokuspokus"
elif out_speaker == "Native German Male":
    speaker_idx = "Karlsson"
elif out_speaker == "Native Japanese Male":
    speaker_idx = "kokoro"
elif out_speaker == "Native Japanese Female":
    speaker_idx = "jp_jsut"
elif out_speaker == "Native Arabic Male":
    speaker_idx = "ar_male"

speed_factor = st.slider("Choose speed factor", min_value=0.5, max_value=2.0, step=0.1, value=1.0)
    
def text_to_speech2(input_language, text, speaker_idx, speed_factor):
    synthesizer_origin = LoadingModels()
    
    file_name = text.replace(" ", "_")[0:20]
    file_name = file_name.translate(
        str.maketrans('', '', string.punctuation.replace('_', ''))) + '.wav'
    wavs, durs, phs, randn_ins = synthesizer_origin.tts(text, speaker_name=speaker_idx, speaker_wav=None, language_name=input_language, style_wav=None, gt_wav=None, speed_factor=speed_factor)
    file_name_origin = input_language+'_'+speaker_idx+'_origin_'+file_name

    synthesizer_origin.save_wav(wavs, f"temp/{file_name_origin}")

    return file_name_origin, durs, phs, randn_ins

def cal_str(ph, dr):
    ph_str = ''
    # print(len(ph_origin))
    # print(len(durs_origin))
    for i, ph in enumerate(ph_origin):
        # for _ in range(int(durs_origin[i])):
        ph_str+=ph
    ph_str = ph_str.replace('<BLNK>','~') 
    return ph_str

# if out_speaker in ["Native Japanese Female"]:
#     flag= True
# else:
#     flag = False
# model1 = st.checkbox('Use the origin model.', disabled=flag) and ~flag
model1 = st.checkbox('Use the origin model.')
# model_middle = st.checkbox('Use the middel model.')
# model2 = st.checkbox('Use the latest model.')
if st.button("Synthesis"): 
    if text is '':
        st.error("Please insert some text.")
    elif not model1:
        st.error("Please select one model.")
    else:
        # result_modified, result_origin = text_to_speech(input_language, text, speaker_idx, speed_factor)
        if model1:
            result_origin, durs_origin, ph_origin, randn_ins = text_to_speech2(input_language, text, speaker_idx, speed_factor)
            audio_file = open(f"temp/{result_origin}", "rb")
            audio_bytes = audio_file.read()
            st.markdown(f"## Your audio with model origin:")
            st.audio(audio_bytes, format="audio/wav", start_time=0)

            # st.text_area('current random seed', randn_ins[0]) 
            txt = cal_str(ph_origin, durs_origin)
            # st.text_area('current duration', txt) 
            st.text_area('phonemes:', txt) 
            
            # text_name = result_origin.replace(".wav", ".txt")
            # if os.path.isfile(f"temp/{text_name}"):
            #     txt_pre = open(f"temp/{text_name}", "r").read()
            #     st.text_area('previous duration', txt_pre)

            # open(f"temp/{text_name}", "w").write(txt)

            # fig, ax = plt.subplots()
            # labels = [str(i) if str(i)!="<BLNK>" else "" for i in ph_origin]
            # # print(randn_ins)
            # # # colors= ['k' if i==178 else 'b' for i in ph_origin]
            # for n in range(len(randn_ins)):
            #     # if n>100:
            #     #     continue
            #     ax.plot(
            #         [i for i in range(len(randn_ins[n]))],
            #         randn_ins[n], 'o-', markersize=2, color='k')
            # # ax.imshow(randn_ins)
            # # ax.set_xticks()
            # ax.set_xticklabels(['', '', ''] + labels)
            # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            # #
            # st.pyplot(fig)
            
        
def remove_files(n, file_type):
    mp3_files = glob(f"temp/*{file_type}")
    if len(mp3_files) != 0:
        now = time.time()
        n_days = n * 86400
        for f in mp3_files:
            if os.stat(f).st_mtime < now - n_days:
                os.remove(f)
                print("Deleted ", f)

remove_files(1, 'wav')
remove_files(1, 'txt')