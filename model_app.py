import joblib
import streamlit as st

model = joblib.load("regression.joblib")

size = st.number_input("Size", min_value=0.1)
nb_rooms = st.number_input("Number of bedrooms", step=1, min_value=1)
garden = st.number_input("Garden", min_value=0, max_value=1, value=0)

if st.button("Predict Price"):
    prediction = model.predict([[size, nb_rooms, garden]])
    st.write(f"Predicted Price: ${prediction[0]:,.2f}")


