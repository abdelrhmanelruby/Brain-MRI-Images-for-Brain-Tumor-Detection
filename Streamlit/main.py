import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
import io
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if 'model' not in st.session_state:
    st.session_state.model = 'Brain Tumor Detection'
def update_radio():
    st.session_state.model =st.session_state.radio

if 'clas' not in st.session_state:
    st.session_state.clas = '2 Classes'
def update_selbox():
    st.session_state.clas =st.session_state.box

if 'check' not in st.session_state:
    st.session_state.check1 = False
def update_check():
    st.session_state.check1 =st.session_state.check

def update_photo():
    st.session_state.photo =st.session_state.image

def pred(img,radio,selbox,check):
    img = tf.keras.utils.load_img(
    img,
    grayscale=False,
    color_mode='rgb',
    target_size=(224,224),
    interpolation='nearest',
    keep_aspect_ratio=False
    )
    os.remove(st.session_state.image.name)
    img = np.array(img).reshape(-1, 224, 224, 3)
    if radio =='Alzheimer Detection':
        model = keras.models.load_model('alzheimer_99.5.h5')
        result=['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
    else:
        if selbox == '44 Classes':
            model = keras.models.load_model('44class_96.5.h5')
            result=['Astrocitoma T1','Astrocitoma T1C+','Astrocitoma T2','Carcinoma T1','Carcinoma T1C+','Carcinoma T2','Ependimoma T1','Ependimoma T1C+','Ependimoma T2','Ganglioglioma T1','Ganglioglioma T1C+',
            'Ganglioglioma T2','Germinoma T1','Germinoma T1C+','Germinoma T2','Glioblastoma T1','Glioblastoma T1C+','Glioblastoma T2','Granuloma T1','Granuloma T1C+','Granuloma T2','Meduloblastoma T1',
            'Meduloblastoma T1C+','Meduloblastoma T2','Meningioma T1','Meningioma T1C+','Meningioma T2','Neurocitoma T1','Neurocitoma T1C+','Neurocitoma T2','Oligodendroglioma T1','Oligodendroglioma T1C+',
            'Oligodendroglioma T2','Papiloma T1','Papiloma T1C+','Papiloma T2','Schwannoma T1','Schwannoma T1C+','Schwannoma T2','Tuberculoma T1','Tuberculoma T1C+','Tuberculoma T2','_NORMAL T1','_NORMAL T2']
        if selbox == '17 Classes':
            model = keras.models.load_model('17class_98.1.h5')
            result=['Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1','Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1C+','Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T2',
            'Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1','Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1C+','Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T2','NORMAL T1','NORMAL T2','Neurocitoma (Central - Intraventricular, Extraventricular) T1','Neurocitoma (Central - Intraventricular, Extraventricular) T1C+',
            'Neurocitoma (Central - Intraventricular, Extraventricular) T2','Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1','Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1C+','Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T2','Schwannoma (Acustico, Vestibular - Trigeminal) T1',
            'Schwannoma (Acustico, Vestibular - Trigeminal) T1C+','Schwannoma (Acustico, Vestibular - Trigeminal) T2']
        if selbox == '15 Classes':
            model = keras.models.load_model('15class_99.8.h5')
            result=['Astrocitoma','Carcinoma','Ependimoma','Ganglioglioma','Germinoma','Glioblastoma','Granuloma','Meduloblastoma','Meningioma','Neurocitoma','Oligodendroglioma','Papiloma','Schwannoma','Tuberculoma','_NORMAL']
        if selbox == '2 Classes':
            model = keras.models.load_model('2calss_lagre_dataset_99.1.h5')
            result=['no', 'yes']
    pred= model.predict(img)
    if check:
        pred=pd.DataFrame({
        'class_name' : result,
        'pred_score' : pred.flatten()
        })
        return pred
    pred = np.argmax(pred, axis=1)
    return result[pred[0]]

def spr_sidebar():
    menu=option_menu(
        menu_title=None,
        options=['Home','About'],
        icons=['house','info-square'],
        menu_icon='cast',
        default_index=0,
        orientation='horizontal'
    )
    if menu=='Home':
        st.session_state.app_mode = 'Home'
    elif menu=='About':
        st.session_state.app_mode = 'About'
    
def home_page():
    st.session_state.check=st.session_state.check1
    st.session_state.radio=st.session_state.model
    st.session_state.box=st.session_state.clas
    if 'photo' in st.session_state:
        st.session_state.image=st.session_state.photo

    st.title('Brain Tumor Detection')
    st.session_state.image=st.file_uploader('Upload MRI Image',accept_multiple_files=False,type=['png', 'jpg','jpeg'],key="upload",on_change=update_photo)
    if st.session_state.image != None:
        st.image(st.session_state.image,width=300)
        radio=st.radio("Model",options=('Brain Tumor Detection','Alzheimer Detection'),key='radio',on_change=update_radio)
        check=st.checkbox('Show Prediction Scores',key='check',on_change=update_check)
        if radio =='Brain Tumor Detection':
            selbox=st.selectbox("choose a number of Classes",options=('44 Classes','17 Classes' ,'15 Classes','2 Classes'),index=0,key='box',on_change=update_selbox)
        else:
            selbox=st.radio("choose a number of Classes",options=(['4 Classes']),index=0,key='box1',on_change=update_selbox)
        state =st.button('Get Result')
        if state:
            f=open(st.session_state.image.name, 'wb') 
            f.write(st.session_state.image.getbuffer())
            f.close()
            
            st.write(pred(st.session_state.image.name,radio,selbox,check))



def About_page():
    st.write("Soon")

def main():
    spr_sidebar()        
    if st.session_state.app_mode == 'Home':
        home_page()
    if st.session_state.app_mode == 'About' :
        About_page()
# Run main()
if __name__ == '__main__':
    main()