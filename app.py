import streamlit as st
import pandas as pd
import numpy as np

csv_path = "processed.csv.gz"
n_cols = 4
n_results = 300

allergens = sorted(
    [
        "",
        "mustard",
        "nuts",
        "pork",
        "yeast",
        "flour",
        "sesame",
        "soy",
        "eggs",
        "gluten",
        "milk",
        "fish",
        "wheat",
    ]
)


@st.cache
# Loading the CSV file takes some time, so we want to cache it
def load_data():
    df = pd.read_csv(csv_path)
    df["allergens"].replace(np.nan, "", regex=True, inplace=True)
    return df


df = load_data()

st.sidebar.title("Open Food Facts Explorer")

cal = st.sidebar.slider("Calories per 100g", 0, 900, (100, 500), 50)
df = df[(df["energy-kcal_100g"] > cal[0]) & (df["energy-kcal_100g"] < cal[1])]

aller = st.sidebar.selectbox("Allergens", allergens)
df = df[df["allergens"].str.lower().str.contains(aller)]

cols = st.beta_columns(n_cols)
df = df[:n_results].reset_index()

for i, row in df.iterrows():
    cols[i % n_cols].image(row["image_small_url"], use_column_width=True)