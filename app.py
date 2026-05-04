import streamlit as st
import pandas as pd
import fitz
from PIL import Image
import pytesseract
import re
from io import BytesIO

st.title("Extraction PDF pressiométrique")

uploaded = st.file_uploader("Importer PDF", type=["pdf"])

def extract_numbers(text):
    return re.findall(r"\d+[.,]\d+|\d+", text)

def clean(v):
    return v.replace(".", ",")

if uploaded:
    doc = fitz.open(stream=uploaded.read(), filetype="pdf")

    data = []

    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(3,3))
        img = Image.open(BytesIO(pix.tobytes()))

        text = pytesseract.image_to_string(img)

        nums = extract_numbers(text)

        for i in range(0, len(nums), 3):
            try:
                data.append({
                    "Profondeur": clean(nums[i]),
                    "Pl": clean(nums[i+1]),
                    "Em": clean(nums[i+2])
                })
            except:
                pass

    df = pd.DataFrame(data)

    st.dataframe(df)

    st.download_button(
        "Télécharger Excel",
        df.to_csv(index=False).encode(),
        "resultat.csv"
    )
