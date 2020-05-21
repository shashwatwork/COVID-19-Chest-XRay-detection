import streamlit as st
import numpy as np
import pandas as pd
from fastai import *
from fastai.vision import *
from fastai.core import *
import os
import time
import io
from PIL import Image
import emoji
import plotly.express as px

MODELSPATH = './models/'
DATAPATH = './data/'
learn = load_learner(MODELSPATH, 'export.pkl')
classes = learn.data.classes


def render_header():
    st.write("""
        <p align="center"> 
            <H1> Skin cancer Analyzer 
        </p>

    """, unsafe_allow_html=True)


@st.cache
def load_mekd():
    img = Image.open(DATAPATH + '/sample.jpeg')
    return img


@st.cache
def load_mekd_():
    img = Image.open(DATAPATH + '/sample2.png')
    return img


def predict_single(img_file):
    'function to take image and return prediction'
    prediction = learn.predict(open_image(img_file))
    probs_list = prediction[2].numpy()
    result = {
        'classes': classes[prediction[1].item()],
        'Probability': {c: round(float(probs_list[i]), 2) for (i, c) in enumerate(classes)}
    }

    result = result.get('classes')

    return result


def main():
    st.sidebar.header('COVID-19 CHEST XRAY DETECTION')

    st.sidebar.subheader('Choose a page to proceed:')
    page = st.sidebar.selectbox("", ["Sample Data", "Upload Your Image"])
    st.write(emoji.emojize('Made with :red_heart: Streamlit ', use_aliases=True))

    if page == "Sample Data":
        st.header("Sample Data for Detecting COVID-19 ")
        st.markdown("""
        You need to choose Sample Data
        """)
        option = st.selectbox('Please Select Sample Data',
                              ('Sample Data I', 'Sample Data II'))
        st.write('You selected:', option)
        if (option == 'Sample Data I'):
            if st.checkbox('Show Sample Data I'):
                st.info("Showing Sample data I---->>>")
                image = load_mekd()
                st.image(image, caption='Sample Data I', use_column_width=True)
                st.subheader("Choose Training Algorithm!")
                if st.checkbox('FastAI'):
                    st.success("Hooray !! Fast AI Model Loaded!")
                    if st.checkbox('Show Prediction  on Sample Data'):
                        img_file = DATAPATH + '/sample.jpeg'
                        result = predict_single(img_file)
                        st.success(
                            'Patient is classified as {} :mask:'.format(result))
        else:
            if st.checkbox('Show Sample Data II'):
                st.info("Showing Sample data II---->>>")
                image = load_mekd_()
                st.image(image, caption='Sample Data II',
                         use_column_width=True)
                st.subheader("Choose Training Algorithm!")
                if st.checkbox('FastAI'):
                    st.success("Hooray !! Fast AI Model Loaded!")
                    if st.checkbox('Show Prediction  on Sample Data'):
                        img_file = DATAPATH + '/sample2.png'
                        result = predict_single(img_file)
                        st.success(
                            'Patient is classified as {} :mask:'.format(result))
    if page == "Upload Your Image":

        st.header("Upload Your Image")

        file_path = st.file_uploader('Upload an image', type=['png', 'jpeg'])

        if file_path is not None:
            image = Image.open(file_path)
            img_array = np.array(image)

            st.success('File Upload Success!!')
        else:
            st.info('Please upload Image file')

        if st.checkbox('Show Uploaded Image'):
            st.info("Showing Uploaded Image ---->>>")
            st.image(img_array, caption='Uploaded Image',
                     use_column_width=True)
            st.subheader("Choose Training Algorithm!")
            if st.checkbox('FAST AI'):
                st.success("Hooray !! FAST AI Model Loaded!")
                if st.checkbox('Show Prediction  for Uploaded Image'):
                    result = predict_single(file_path)
                    st.success(
                        'Patient is classified as {} :mask:'.format(result))


if __name__ == "__main__":
    main()
