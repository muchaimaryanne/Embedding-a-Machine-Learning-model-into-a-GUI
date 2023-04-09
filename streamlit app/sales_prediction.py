import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np
import joblib,os
from PIL import Image

st.set_page_config(page_title="Sales Prediction app", page_icon="icon.png", layout="centered")
DIRPATH = os.path.dirname(os.path.realpath(__file__))

model = pickle.load(open('components.pickle', 'rb'))

st.title('Store Sales Prediction')
st.sidebar.header('Sales Data')
image = Image.open('ss.jpg')
st.image(image, '')

# FUNCTION
def user_report():
  #date = st.date_input("date")
  onpromotion = st.sidebar.slider('onpromotion', 50,100, 1 )
  store_nbr = st.sidebar.slider('store_nbr', 0,100, 1 )
  cluster = st.sidebar.slider('cluster', 0,30, 1 )
  dcoilwtico = st.sidebar.slider('dcoilwtico', 0,10, 1 )
  type_y_NA = st.sidebar.slider('type_y_N/A', 0,3, 1 )



  user_report_data = {
      #'date':date,
      'onpromotion':onpromotion,
      'store_nbr':store_nbr,
      'cluster':cluster,
      'dcoilwtico':dcoilwtico,
      'type_y_N/A':type_y_NA
   
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

user_data = user_report()
st.header('Sales Data')
st.write(user_data)

sales = model['model'].predict(user_data)
st.subheader('Store Sales')
st.subheader('$'+str(np.round(sales[0], 2)))