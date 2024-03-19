# app.py
import streamlit as st
from main import predict_disease

def app():
    st.title('Skin Disease Classification App')

    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
        st.write(file_details)

        if st.button('Predict Disease'):
            result = predict_disease(uploaded_file)
            st.write(f"The predicted disease for the given image is: {result}")

if __name__ == '__main__':
    app()