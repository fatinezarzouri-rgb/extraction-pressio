import streamlit as st
import pandas as pd
import fitz
import re
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Extraction pressiométrique", layout="wide")
st.title("Préparation Excel pressiométrique")

uploaded = st.file_uploader("Importer le PDF", type=["pdf"])

DEPTHS = [1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12, 13.5, 15]

def fr(x):
    return str(x).replace(".", ",")

def make_excel(df):
    out = BytesIO()

    export = df[
        [
            "Nom du sondages",
            "x",
            "y",
            "Profondeur (m)",
            "Lithologie",
            "Pl* (MPa)",
            "Em (MPa)"
        ]
    ]

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        export.to_excel(writer, index=False, startrow=1, sheet_name="Pressiometrique")
        ws = writer.book["Pressiometrique"]

        ws.merge_cells("A1:D1")
        ws["A1"] = "Echantillon"

        ws.merge_cells("E1:G1")
        ws["E1"] = "Caracteristiques pressiometriques"

    out.seek(0)
    return out

if uploaded:
    doc = fitz.open(stream=uploaded.read(), filetype="pdf")

    rows = []

    for page_index in range(len(doc)):
        sondage = f"Sondage_page_{page_index + 1}"

        for i, depth in enumerate(DEPTHS):
            rows.append({
                "Nom du sondages": sondage if i == 0 else "",
                "x": "",
                "y": "",
                "Profondeur (m)": fr(depth),
                "Lithologie": "",
                "Pl* (MPa)": "",
                "Em (MPa)": ""
            })

    df = pd.DataFrame(rows)

    st.subheader("Complète / corrige le tableau avant export")

    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="dynamic"
    )

    excel = make_excel(edited_df)

    st.download_button(
        "Télécharger Excel final",
        data=excel,
        file_name="extraction_pressiometrique.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
