import streamlit as st
import pandas as pd
from iris import IRIS
from PIL import Image

iris_path = "iris_210118090818.feather"


@st.cache
# Loading the CSV file takes some time, so we want to cache it
def load_data():
    # df = pd.read_csv(csv_path)
    return IRIS(iris_path)


iris = load_data()

st.title("Rate my Food")

f = st.file_uploader("Select Image")

if f is not None:
    img = Image.open(f)
    if img.size[0] > img.size[1]:
        img = img.rotate(270)
    # st.header("Uploaded Image")
    # st.image(img)
    id = iris.search(img)[0]
    # st.write(iris.meta.loc[id])
    brand = iris.meta.loc[id, "brands"]
    product = iris.meta.loc[id, "product_name"]
    st.header(brand + " - " + product)
    cols = st.beta_columns(2)
    row = iris.meta.loc[id]
    cols[0].image(img, use_column_width=True)
    cols[1].text("Calories: " + str(round(row["energy-kcal_100g"], 1)))
    cols[1].text("Protein: " + str(round(row["proteins_100g"], 1)) + "g")
    cols[1].text("Carbs: " + str(round(row["carbohydrates_100g"], 1)) + "g")
    cols[1].text("Fat: " + str(round(row["fat_100g"], 1)) + "g")
    cols[1].text("Sugar: " + str(round(row["sugars_100g"], 1)) + "g")
    cols[1].text("Salt: " + str(round(row["salt_100g"], 3)) + "g")
    cols[1].image(
        Image.open("assets/" + row["nutriscore_grade"] + ".png"),
        use_column_width=True,
    )
