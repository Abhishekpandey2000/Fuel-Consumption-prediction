import numpy as np
import pickle as pk
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

loaded_model = pk.load(
    open("trained_model.sav", "rb"))


def hpp(input_data):
    # input_data = (4.98, 2.31, 0.538, 15.3, 6.575, 296, 4.09, 65)
    arr = np.asarray(input_data)
    arr = arr.reshape(1, -1)
    ar = sc.fit_transform(arr)

    prediction = loaded_model.predict(ar)
    return(f"The Median value of owner-occupied homes in $1000's is {round(prediction[0],2)}")



def main():
    # giving a title
    st.title("Housing Price Prediction System")

    # getting the input data from user
    result = 0

    lstat = st.text_input("% of lower class population around")
    indus = st.text_input("proportion of non-retail business acres per town")
    nox = st.text_input("nitric oxides concentration (parts per 10 million)")
    ptratio = st.text_input("pupil-teacher ratio by town")
    rm = st.text_input("average number of rooms per dwelling")
    tax = st.text_input("full-value property-tax rate per $10,000")
    dis = st.text_input("weighted distances to five city employment centres")
    age = st.text_input("proportion of owner-occupied units built prior to 1940")

    # code for prediction

    # creating a button for prediction
    if st.button("Predict MEDV"):
        result = hpp([lstat, indus, nox, ptratio, rm, tax, dis, age])

    st.success(result)


if __name__ == "__main__":
    main()