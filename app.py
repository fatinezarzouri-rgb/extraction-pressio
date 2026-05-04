import streamlit as st
import pandas as pd
import fitz
from PIL import Image
import pytesseract
import re
from io import BytesIO

# 🔥 IMPORTANT pour Streamlit Cloud
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

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
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
        img = Image.open(BytesIO(pix.tobytes()))

        # 🔥 amélioration OCR
        text = pytesseract.image_to_string(img, lang="fra+eng")

        nums = extract_numbers(text)

        # 🔥 filtrage pour éviter n'importe quoi
        nums = [n for n in nums if float(n.replace(",", ".")) < 10000]

        # 🔥 regroupement correct (Profondeur / Pl / Em)
        for i in range(0, len(nums) - 2, 3):
            try:
                profondeur = clean(nums[i])
                pl = clean(nums[i + 1])
                em = clean(nums[i + 2])

                data.append({
                    "Profondeur (m)": profondeur,
                    "Pl (MPa)": pl,
                    "Em (MPa)": em
                })
            except:
                pass

    df = pd.DataFrame(data)

    st.subheader("Résultat")
    st.dataframe(df, use_container_width=True)

    # 🔥 export Excel (pas CSV)
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)

    st.download_button(
        "Télécharger Excel",
        data=output.getvalue(),
        file_name="resultat.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
