import streamlit as st
import pandas as pd
import fitz
import base64
import json
from PIL import Image
from io import BytesIO
from openai import OpenAI

st.set_page_config(page_title="Extraction Vision AI", layout="wide")
st.title("Extraction pressiométrique par Vision AI")

api_key = st.text_input("OpenAI API Key", type="password")
uploaded = st.file_uploader("Importer PDF", type=["pdf"])

PROMPT = """
Tu es un expert en extraction de coupes pressiométriques géotechniques.

Lis cette page comme une image technique.

Objectif: extraire un tableau JSON strict.

Règles:
- Lire le nom du sondage en haut.
- Lire X et Y en haut.
- Lire les valeurs Pl* en bleu.
- Lire les valeurs Em en rouge dans la colonne Em uniquement.
- Ne pas prendre Pf.
- Lire la profondeur de chaque point selon son niveau horizontal sur l'axe vertical.
- Associer chaque profondeur à la lithologie correspondante selon les couches dessinées à gauche.
- Retourner uniquement du JSON valide, sans explication.

Format JSON:
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

def extract_page(client, page):
    b64 = page_to_base64(page)

    response = client.responses.create(
        model="gpt-5.4-mini",
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

    text = response.output_text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)

def make_excel(df):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, startrow=1, sheet_name="Pressiometrique")
        ws = writer.book["Pressiometrique"]

        ws.merge_cells("A1:D1")
        ws["A1"] = "Echantillon"

        ws.merge_cells("E1:G1")
        ws["E1"] = "Caracteristiques pressiometriques"

    output.seek(0)
    return output

if uploaded and api_key:
    if st.button("Extraire Excel avec Vision AI"):
        client = OpenAI(api_key=api_key)
        doc = fitz.open(stream=uploaded.read(), filetype="pdf")

        all_rows = []
        progress = st.progress(0)

        for i, page in enumerate(doc):
            rows = extract_page(client, page)
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
