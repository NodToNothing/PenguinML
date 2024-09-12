import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("Penguin Classifer with Upload")
penguin_file = st.file_uploader("Upload your penguin data")

if penguin_file is None:
    rf_pickle = open("random_forest_penguin.pickle", 'rb')
    map_pickle = open("output_penguin.pickle", 'rb')

    rfc = pickle.load(rf_pickle)
    unique_penguin_mapping = pickle.load(map_pickle)

    rf_pickle.close()
    map_pickle.close()

else:
    penguin_df = pd.read_csv("nautilus.csv")
    # print(penguin_df.head())

    penguin_df.dropna(inplace=True)
    output = penguin_df["species"]

    features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]

    #one-hot encoding
    features = pd.get_dummies(features)
    print(features) #see all the resulting numericized variables (lots of bools for categories)

    # output = 0/1/2/1 or similar
    # uniques = "a", "b", "c" - no duplicates
    output, unique_penguin_mapping = pd.factorize(output)

    #reserve part of the data
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size = .8)

    rfc = RandomForestClassifier(random_state = 15) #this is a seed to maintain same set, geeks use 42
    rfc.fit (x_train.values, y_train) #build the forest from x/y

    y_pred = rfc.predict(x_test.values) #use the test values to generate an accuracy score
    score = round(accuracy_score(y_pred, y_test),2)

    print("Our accuracy score for this model is {}".format(score))

with st.form("User Inputs"):
    island = st.selectbox("Penguin Island", options=["Biscoe", "Dream", "Torgerson"])
    sex = st. selectbox("Sex", options=["Female", "Male"])
    bill_length = st.number_input("Bill Length in mm", min_value = 0)
    bill_depth = st.number_input("Bill depth in mm", min_value = 0)
    flipper_length = st.number_input("Flipper length in mm", min_value=0)
    body_mass = st.number_input("Body mass in g", min_value=0)
    st.form_submit_button()


island_biscoe, island_dream, island_torgerson = 0,0,0
if island == "Biscoe":
    island_biscoe = 1
elif island == "Dream":
    island_dream = 1
elif island == "Torgerson":
    island_torgerson = 1

sex_female, sex_male = 0, 0
if sex == "Female":
    sex_female = 1
elif sex == "Male":
    sex_male = 1

user_inputs = [island, sex, bill_length, bill_depth, flipper_length, body_mass]

new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length, body_mass, island_biscoe, island_dream, island_torgerson, sex_female, sex_male]])
prediction_species = unique_penguin_mapping[new_prediction][0]
st.write(f"""The user inputs are {user_inputs}""".format())

st.write(f"Your penguin prediction is: {prediction_species}")

st.write(rfc)
st.write(unique_penguin_mapping)