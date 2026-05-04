import streamlit as st
import pandas as pd
import fitz
import pytesseract
import cv2
import numpy as np
import re
from PIL import Image
from io import BytesIO
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.set_page_config(page_title="Extraction pressiométrique", layout="wide")
st.title("Extraction automatique des coupes pressiométriques vers Excel")

uploaded = st.file_uploader("Importer le PDF", type=["pdf"])

# Profondeurs standards des essais pressiométriques
PROF_STD = [1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12, 13.5, 15]


def fr_num(x):
    if x is None or x == "":
        return ""
    return str(x).replace(".", ",")


def to_float(x):
    try:
        return float(str(x).replace(",", "."))
    except:
        return None


def ocr_data(img, psm=6):
    return pytesseract.image_to_data(
        img,
        lang="fra+eng",
        config=f"--psm {psm}",
        output_type=pytesseract.Output.DATAFRAME
    )


def crop(img, box):
    w, h = img.size
    x1, y1, x2, y2 = box
    return img.crop((int(w*x1), int(h*y1), int(w*x2), int(h*y2)))


def extract_header(img):
    header = crop(img, (0.33, 0.13, 0.97, 0.25))
    text = pytesseract.image_to_string(header, lang="fra+eng", config="--psm 6")

    sondage = ""
    x = ""
    y = ""

    m = re.search(r"Sondage\s*[:\-]?\s*([A-Z]{1,4}[_\-]?[A-Za-z0-9]+)", text, re.I)
    if m:
        sondage = m.group(1).strip()

    mx = re.search(r"X\s*[:\-]?\s*([\d\s.,]+)", text)
    my = re.search(r"Y\s*[:\-]?\s*([\d\s.,]+)", text)

    if mx:
        x = mx.group(1).strip().replace(" ", "")
    if my:
        y = my.group(1).strip().replace(" ", "")

    return sondage, x, y


def detect_layer_lines(litho_img):
    arr = np.array(litho_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=int(arr.shape[1] * 0.45),
        maxLineGap=8
    )

    ys = [0, arr.shape[0]]

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) <= 3:
                ys.append(int((y1 + y2) / 2))

    ys = sorted(list(set(ys)))

    clean = []
    for y in ys:
        if not clean or abs(y - clean[-1]) > 15:
            clean.append(y)

    return clean


def extract_lithology_texts(litho_img):
    df = ocr_data(litho_img, psm=6)
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip() != ""]

    words = []
    for _, r in df.iterrows():
        txt = str(r["text"]).strip()
        if re.search(r"[A-Za-zéèêàùçîôû]", txt):
            cy = r["top"] + r["height"] / 2
            words.append((cy, txt))

    words = sorted(words, key=lambda x: x[0])

    groups = []
    for cy, txt in words:
        if not groups or abs(cy - groups[-1]["cy"]) > 25:
            groups.append({"cy": cy, "words": [txt]})
        else:
            groups[-1]["words"].append(txt)
            groups[-1]["cy"] = (groups[-1]["cy"] + cy) / 2

    lithos = []
    ignore = ["Lithologie", "Profondeur", "LABOTEST"]
    for g in groups:
        t = " ".join(g["words"])
        if not any(i.lower() in t.lower() for i in ignore):
            lithos.append({"cy": g["cy"], "text": t})

    return lithos


def build_layers(litho_img, max_depth=15):
    lines = detect_layer_lines(litho_img)
    texts = extract_lithology_texts(litho_img)

    h = litho_img.size[1]
    layers = []

    for i in range(len(lines) - 1):
        y1 = lines[i]
        y2 = lines[i + 1]
        mid = (y1 + y2) / 2

        z1 = round((y1 / h) * max_depth, 2)
        z2 = round((y2 / h) * max_depth, 2)

        litho = ""
        if texts:
            nearest = min(texts, key=lambda t: abs(t["cy"] - mid))
            litho = nearest["text"]

        if z2 > z1:
            layers.append({
                "z_debut": z1,
                "z_fin": z2,
                "lithologie": litho
            })

    return layers


def get_lithology(depth, layers):
    d = to_float(depth)
    if d is None:
        return ""

    for c in layers:
        if c["z_debut"] <= d <= c["z_fin"]:
            return c["lithologie"]

    return ""


def extract_zone_numbers(zone_img):
    df = ocr_data(zone_img, psm=6)
    df = df.dropna(subset=["text"])

    values = []

    for _, r in df.iterrows():
        txt = str(r["text"]).strip()
        txt = txt.replace("O", "0").replace("o", "0")
        txt = txt.replace("l", "1")

        m = re.search(r"\d+[.,]\d+|\d+", txt)
        if m:
            val = m.group(0)
            f = to_float(val)
            if f is not None:
                cy = r["top"] + r["height"] / 2
                values.append({"y": cy, "value": f})

    values = sorted(values, key=lambda x: x["y"])

    clean = []
    for v in values:
        if not clean or abs(v["y"] - clean[-1]["y"]) > 18:
            clean.append(v)

    return clean


def match_by_order(values, limit=10):
    values = values[:limit]
    return values


def process_page(img, page_num):
    sondage, x, y = extract_header(img)

    litho_img = crop(img, (0.13, 0.28, 0.39, 0.90))
    pl_img = crop(img, (0.63, 0.28, 0.78, 0.90))
    em_img = crop(img, (0.78, 0.28, 0.96, 0.90))

    layers = build_layers(litho_img, max_depth=15)

    pl_vals = match_by_order(extract_zone_numbers(pl_img), 10)
    em_vals = match_by_order(extract_zone_numbers(em_img), 10)

    rows = []
    n = min(len(PROF_STD), max(len(pl_vals), len(em_vals)))

    for i in range(n):
        prof = PROF_STD[i]

        rows.append({
            "Page PDF": page_num,
            "Nom du sondages": sondage,
            "x": x if i == 0 else "",
            "y": y if i == 0 else "",
            "Profondeur (m)": fr_num(prof),
            "Lithologie": get_lithology(prof, layers),
            "Pl* (MPa)": fr_num(round(pl_vals[i]["value"], 3)) if i < len(pl_vals) else "",
            "Em (MPa)": fr_num(round(em_vals[i]["value"], 1)) if i < len(em_vals) else ""
        })

    return rows, layers


def make_excel(df):
    output = BytesIO()

    export_df = df[
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

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, startrow=1, sheet_name="Pressiometrique")
        ws = writer.book["Pressiometrique"]

        ws.merge_cells("A1:D1")
        ws["A1"] = "Echantillon"
        ws.merge_cells("E1:G1")
        ws["E1"] = "Caracteristiques pressiometriques"

        blue = PatternFill("solid", fgColor="7EC8E3")
        thin = Side(style="thin", color="000000")
        border = Border(left=thin, right=thin, top=thin, bottom=thin)

        for row in ws.iter_rows():
            for cell in row:
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                cell.border = border

        for cell in ws[1]:
            cell.fill = blue
            cell.font = Font(bold=True)

        for cell in ws[2]:
            cell.fill = blue
            cell.font = Font(bold=True)

        widths = {
            "A": 18, "B": 15, "C": 15, "D": 15,
            "E": 35, "F": 14, "G": 14
        }

        for col, width in widths.items():
            ws.column_dimensions[col].width = width

    output.seek(0)
    return output


if uploaded:
    with st.spinner("Analyse du PDF en cours..."):
        doc = fitz.open(stream=uploaded.read(), filetype="pdf")

        all_rows = []
        debug_layers = []

        for page_num, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            img = Image.open(BytesIO(pix.tobytes("png")))

            rows, layers = process_page(img, page_num)
            all_rows.extend(rows)

            for l in layers:
                l["Page PDF"] = page_num
                debug_layers.append(l)

        df = pd.DataFrame(all_rows)

    st.subheader("Résultat extrait")
    st.dataframe(df, use_container_width=True)

    with st.expander("Voir les couches détectées"):
        st.dataframe(pd.DataFrame(debug_layers), use_container_width=True)

    excel = make_excel(df)

    st.download_button(
        "Télécharger Excel",
        excel,
        file_name="extraction_pressiometrique.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
