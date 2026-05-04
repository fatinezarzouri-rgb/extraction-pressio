import streamlit as st
import pandas as pd
import fitz
import base64
import json
from io import BytesIO
from openai import OpenAI

st.set_page_config(page_title="Extraction Vision AI", layout="wide")
st.title("Extraction pressiométrique par Vision AI")

uploaded = st.file_uploader("Importer PDF", type=["pdf"])

api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

PROMPT = """
Tu es un expert en extraction de coupes pressiométriques géotechniques.

Analyse cette page comme une image technique.

Objectif : extraire un tableau propre.

Règles :
- Lire le nom du sondage en haut, exemple SP_Reta043.
- Lire X et Y en haut.
- Lire Pl* en bleu.
- Lire Em en rouge uniquement dans la colonne Em.
- Ne jamais prendre Pf.
- Lire la profondeur depuis l’axe vertical.
- Associer chaque profondeur à la bonne lithologie selon les couches dessinées à gauche.
- Utiliser la virgule comme séparateur décimal.
- Retourner uniquement du JSON valide.

Format :
[
  {
    "Nom du sondages": "",
    "x": "",
    "y": "",
    "Profondeur (m)": "",
    "Lithologie": "",
    "Pl* (MPa)": "",
    "Em (MPa)": ""
  }
]
"""

def page_to_base64(page):
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
    img_bytes = pix.tobytes("png")
    return base64.b64encode(img_bytes).decode("utf-8")

def extract_page(page):
    b64 = page_to_base64(page)

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{b64}"
                    }
                ]
            }
        ]
    )

    txt = response.output_text.strip()
    txt = txt.replace("```json", "").replace("```", "").strip()
    return json.loads(txt)

def make_excel(df):
    out = BytesIO()

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, startrow=1, sheet_name="Pressiometrique")
        ws = writer.book["Pressiometrique"]

        ws.merge_cells("A1:D1")
        ws["A1"] = "Echantillon"

        ws.merge_cells("E1:G1")
        ws["E1"] = "Caracteristiques pressiometriques"

    out.seek(0)
    return out

if uploaded:
    if st.button("Extraire Excel"):
        doc = fitz.open(stream=uploaded.read(), filetype="pdf")

        all_rows = []
        progress = st.progress(0)

        for i, page in enumerate(doc):
            rows = extract_page(page)
            all_rows.extend(rows)
            progress.progress((i + 1) / len(doc))

        df = pd.DataFrame(all_rows)

        st.dataframe(df, use_container_width=True)

        excel = make_excel(df)

        st.download_button(
            "Télécharger Excel",
            excel,
            "extraction_pressiometrique_ai.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
