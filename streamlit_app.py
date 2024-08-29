import streamlit as st
import pandas as pd
import pickle

st.title('Machine learning app')

st.info('This is a Machine learning app!')

with st.expander("Data"):
    st.write("**Raw Data**")
    df = pd.read_csv(
        "https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")
    df

    X = df.drop("species", axis=1)
    y = df.species

with st.expander("Data visualization"):
    # bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g",
    st.scatter_chart(data=df, x="bill_length_mm",
                     y="bill_depth_mm", color="species")

with st.sidebar:
    st.header("Input features")
    # "island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"
    island = st.selectbox("Island", (X.island.unique()))
    sex = st.selectbox("Sex", ("male", "female"))
    bill_length_mm = st.slider(
        "Bill length (mm)", X.bill_length_mm.min(), max_value=X.bill_length_mm.max())

    bill_depth_mm = st.slider(
        "Bill depth (mm)", X.bill_depth_mm.min(), max_value=X.bill_depth_mm.max())

    flipper_length_mm = st.slider(
        "Flipper length (mm)", X.flipper_length_mm.min(), X.flipper_length_mm.max())

    body_mass_g = st.slider(
        "Body mass (g)", X.body_mass_g.min(), X.body_mass_g.max())

    data = {
        "island": island, "bill_length_mm": bill_length_mm, "bill_depth_mm": bill_depth_mm,
        "flipper_length_mm": flipper_length_mm, "body_mass_g": body_mass_g, "sex": sex
    }
with st.expander("input features"):
    input_df = pd.DataFrame(data, index=[0])
    input_df
    # all_pinguins = pd.concat([pd.DataFrame(data, index=[0]), X])

    # categorical_cols = ["island", "sex"]
    # prep_data = pd.get_dummies(all_pinguins, prefix=categorical_cols)
    # st.write("**Encoded input penguins**")
    # prep_data.loc[0, :]

    with open("model.pkl", "rb") as file:
        processing_pipeline = pickle.load(file)

    pred = processing_pipeline.predict(input_df)[0]
    st.success(f"The penguin is classified as **{pred}**")
