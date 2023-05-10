import streamlit as st
import pandas as pd
import numpy as np
import pickle
import librosa
import csv
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def main():
    page_bg_img = '''
    <style>
    body
    {

        opacity: 0.7;

    background: url("https://images.unsplash.com/photo-1536782365487-94eaef48851d?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1770&q=80");
    background - position: center;
    background - size: 100 %;

    }
    < / style >
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)


    st.title("Music Genre Classification System ")
    st.markdown("""<hr style="height:2px;border:solid #FFFFFF;color:#FFFFFF;background-color:#333;" /> """, unsafe_allow_html=True)

    about()



def about():

    st.write('Music Experts have been trying for a long time to understand sound and what differenciates one song from\
         another. How to visualize sound. What makes a tone different from another. This data hopefully can give the\
          opportunity to do just that.')
    st.write('In this Project we will use the GTZAN Dataset which  has a Raw data of 1.2GB and consists of 1000 audio\
         files(.au) divided into 10 folders for 10 genres equally. I.e Every genre has 100 audio files. Different genres are:\
          blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock. Each audio file is in the WAV format\
           and has a sampling rate of 22,050 Hz and a bit depth of 16 bits.')
    st.write("Github Link: [LINK]")

if __name__ == main:
    main()