import streamlit as st
import pandas as pd
import requests
import os
from io import StringIO
import datetime

# Set page configuration
st.set_page_config(page_title="Predict", page_icon="🔮",  layout="wide")


st.title("Predict Sepsis")


def select_model():
    col1, col2 = st.columns(2)
    with col1:
        choice = st.selectbox('Select a model', options=['xgboost', 'random_forest'], key='select_model')
    with col2:
        pass

    # if st.session_state['selected_model'] == 'Xgboost':
    #     pipeline = xgboost_pipeline()
    # else:
    #     pipeline = load_random_forest_pipeline()

    # encoder = joblib.load('models/encoder.joblib')
    return choice


def make_prediction(): 
    selected_model = st.session_state['select_model']
    age = st.session_state['age']
    insurance = 1 if st.session_state['insurance'] == 'Yes' else 0
    m11 = st.session_state['m11']
    pr = st.session_state['pr']
    prg = st.session_state['prg']
    ts = st.session_state['ts']
    pl = st.session_state['pl']
    sk = st.session_state['sk']
    bd2 = st.session_state['bd2']

    base_url = 'https://gabcares-team-curium.hf.space/'
    url =  base_url + f"{ 'xgboost_prediction' if selected_model=='xgboost' else 'random_forest_prediction'}"


    data = {'PRG': prg, 'PL': pl, 'PR': pr,'SK': sk, 'TS': ts, 'M11': m11, 'BD2': bd2, 'Age': age, 'Insurance': insurance}
    # Send POST request with JSON data using the json parameter
    response_status = requests.get(base_url)

    if (response_status.status_code == 200):
        response = requests.post(url, json=data,timeout=30)
        pred_prob = (response.json()['result'])
        prediction = pred_prob['prediction']
        probability = pred_prob['probability']

        st.session_state['prediction'] = prediction
        st.session_state['probability'] = probability
    else:
        st.write('Unable to connect to the server.')


# Creating the form
def display_form():
    select_model()

    with st.form('input_features'):

        col1, col2 = st.columns(2)
        with col1:
            st.write('### Patient Demographics')
            age = st.number_input('Age', min_value=0, max_value=100, step=1, key = 'age')
            insurance = st.selectbox('Insurance', options = ['Yes', 'No'], key = 'insurance')

            st.write('### Vital Signs')
            m11 = st.number_input('BMI', min_value=10.0, format="%.2f",step = 1.00, key = 'm11')
            pr = st.number_input('Blood Pressure', min_value=10.0, format="%.2f",step = 1.00, key = 'pr')
            prg = st.number_input('PRG(plasma glucose)', min_value=10.0, format="%.2f",step = 1.00, key = 'prg')

        with col2:
            st.write('### Blood Work')
            pl = st.number_input('PL(Blood Work Result 1)', min_value=10.0, format="%.2f",step = 1.00, key = 'pl')
            sk = st.number_input('SK(Blood Work Result 2)', min_value=10.0, format="%.2f",step = 1.00, key = 'sk')
            ts = st.number_input('TS(Blood Work Result 3)', min_value=10.0, format="%.2f",step = 1.00, key = 'ts')
            bd2 = st.number_input('BD2(Blood Work Result 4)', min_value=10.0, format="%.2f",step = 1.00, key = 'bd2')
        
        st.form_submit_button('Submit', on_click=make_prediction)
    

if __name__ == '__main__':

    display_form()

    final_prediction = st.session_state.get('prediction')
    final_probability = st.session_state.get('probability')

    if final_prediction is None:
        st.write('Predictions show here!')
        st.divider()
    else:
        if final_prediction.lower() == 'positive':
            st.markdown(f'### Patient is likely to develop sepsis😞.')
            st.markdown(f'## Probability: {final_probability:.2f}%')
            
        else:
            # st.markdown(f'## Sepsis: {final_prediction}')
            st.markdown(f'### Patient is unlikely to develop sepsis😊.')
            st.markdown(f'## Probability: {final_probability:.2f}%')