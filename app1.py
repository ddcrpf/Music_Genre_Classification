import streamlit as st
import pandas as pd
import numpy as np
import pickle
import librosa
import csv
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
#from audio_recorder_streamlit import audio_recorder


# def predict_age(audio_bytes):
#     input = librosa.core.load(audio_bytes, sr=22050, mono=True, offset=0.0, duration=None, dtype="np.float32", res_type='kaiser_best')
#
#     prediction = model.predict(input)
#
#     return prediction


# st.title("Music Genre Classification System ")
# st.text("This ML model classifies the given input song from one of the ten genres<br> - rock, classical, metal,"
#         " disco, blues, reggae, country, hiphop, jazz, pop")
#
# custom_css = "<p>Play an audio of 30 seconds to predict the genre</p>"
# st.markdown(custom_css, unsafe_allow_html=True)


# def save_file(sound_file):
#     # save your sound file in the right folder by following the path
#     with open(os.path.join('audio_files/', sound_file.name), 'wb') as f:
#         f.write(sound_file.getbuffer())
#     return sound_file.name


def transform_wav_to_csv(sound_saved):
    # define the column names
    header_test = 'filename length chroma_stft_mean chroma_stft_var rms_mean rms_var spectral_centroid_mean ' \
                  'spectral_centroid_var spectral_bandwidth_mean \ spectral_bandwidth_var rolloff_mean rolloff_var ' \
                  'zero_crossing_rate_mean zero_crossing_rate_var harmony_mean harmony_var perceptr_mean perceptr_var ' \
                  'tempo mfcc1_mean mfcc1_var mfcc2_mean \ mfcc2_var mfcc3_mean mfcc3_var mfcc4_mean mfcc4_var ' \
                  'mfcc5_mean mfcc5_var mfcc6_mean mfcc6_var mfcc7_mean mfcc7_var mfcc8_mean mfcc8_var mfcc9_mean ' \
                  'mfcc9_var mfcc10_mean mfcc10_var\ mfcc11_mean mfcc11_var mfcc12_mean mfcc12_var mfcc13_mean ' \
                  'mfcc13_var mfcc14_mean mfcc14_var mfcc15_mean mfcc15_var\ mfcc16_mean mfcc16_var mfcc17_mean ' \
                  'mfcc17_var mfcc18_mean  mfcc18_var mfcc19_mean mfcc19_var mfcc20_mean mfcc20_var'.split()

    # create the csv file
    file = open(f'csv_files/{os.path.splitext(sound_saved)[0]}.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header_test)
    # calculate the value of the librosa parameters
    sound_name = f'audio_files/{sound_saved}'
    y, sr = librosa.load(sound_name, mono=True, duration=30)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=27)
    to_append = f'{os.path.basename(sound_name)} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for e in mfcc:
        to_append += f' {np.mean(e)}'
        to_append += f' {np.var(e)}'

    # fill in the csv file
    file = open(f'csv_files/{os.path.splitext(sound_saved)[0]}.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())
    # create test dataframe
    df_test = pd.read_csv(f'csv_files/{os.path.splitext(sound_saved)[0]}.csv')
    # each time you add a sound, a line is added to the test.csv file
    # if you want to display the whole dataframe, you can deselect the following line
    # st.write(df_test)
    st.write(df_test)

    return df_test


def classification(dataframe):
    # create a dataframe with the csv file of the data used for training and validation
    df = pd.read_csv('csv_files/data.csv')
    # OUTPUT: labels => last column
    labels_list = df.iloc[:, -1]

    st.write("label list shpe:", labels_list.shape)
    st.write("label list = ", labels_list)
    # encode the labels (0 => 44)
    converter = LabelEncoder()
    y = converter.fit_transform(labels_list)
    # st.write("y shape:", y.shape)
    # st.write("y = ", y)
    # y = y.reshape(-1, 1)
    # st.write("y shape:", y.shape)
    # INPUTS: all other columns are inputs except the filename
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(df.iloc[:, 1:59]))
    X_test = scaler.transform(np.array(dataframe.iloc[:, 1:59]))

    # load the pretrained model
    model = pickle.load(open("ModelforPrediction.pkl", "rb"))
    # generate predictions for test samples
    predictions = model.predict(X_test)
    # st.write("X-test =", X_test)
    # st.write("X-test.shape =", X_test.shape)
    # st.write("X-test.ndim =", X_test.ndim)

    st.write("predictions =", predictions)
    st.write("predictions.shape =", predictions.shape)
    st.write("predictions.ndim =", predictions.ndim)
    # # generate argmax for predictions
    # classes = np.argmax(predictions, axis = 1)
    # st.write("classes = ", classes)
    # st.write("classes.shape = ", classes.shape)
    # # transform class number into class name
    # result = converter.inverse_transform(classes)
    result = predictions

    # -------------------------------------------------------------------------

    # -------------------------------------------------

    return result


def choice_prediction():
    st.write('# Prediction')
    st.write('### Choose a audio file in .wav format')
    # upload sound
    uploaded_file = st.file_uploader(' ', type='wav')

    if uploaded_file is not None:
        # view details
        file_details = {'filename': uploaded_file.name, 'filetype': uploaded_file.type, 'filesize': uploaded_file.size}
        st.write(file_details)
        # read and play the audio file
        st.write('### Play audio')
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav')
        # save_file function
        # save_file(uploaded_file)
        # define the filename
        sound = uploaded_file.name
        # transform_wav_to_csv function
        transform_wav_to_csv(sound)
        st.write('### Classification results')
        # if you select the predict button
        if st.button('Predict'):
            # write the prediction: the prediction of the last sound sent corresponds to the first column
            st.write("Genre  is: ",
                     str(classification(transform_wav_to_csv(sound))).replace('[', '').replace(']', '').replace("'",'').replace('"', ''))
    else:
        st.write('The file has not been uploaded yet')
    return


def main():
    # st.image(Image.open('logo_ovh.png'), width=200)
    model = pickle.load(open("ModelforPrediction.pkl", "rb"))

    import base64

    page_bg_img = '''
        <style>
        body{
        opacity: 0.6;
        background: url("https://images.unsplash.com/photo-1605731414532-6b26976cc153?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1770&q=80");
        background-position: bottom;
        background-size : 100%;
        }
        </style>
        '''

    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.title("Music Genre Classification System ")
    st.markdown("""<hr style="height:2px;border:solid #FFFFFF;color:#FFFFFF;background-color:#333;" /> """,
                unsafe_allow_html=True)

    choice_prediction()


if __name__ == "__main__":
    main()
