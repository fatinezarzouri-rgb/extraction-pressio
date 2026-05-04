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
st.title("Extraction PDF pressiométrique vers Excel")

uploaded = st.file_uploader("Importer le PDF", type=["pdf"])

DEPTH_MAX = 15.0

HEADER_BOX = (0.30, 0.13, 0.98, 0.25)
LITHO_BOX = (0.18, 0.28, 0.40, 0.90)
PL_BOX = (0.64, 0.28, 0.78, 0.90)
EM_BOX = (0.82, 0.28, 0.97, 0.90)


def crop(img, box):
    w, h = img.size
    return img.crop((int(w*box[0]), int(h*box[1]), int(w*box[2]), int(h*box[3])))


def fr(x):
    if x == "" or x is None:
        return ""
    return str(x).replace(".", ",")


def to_float(x):
    try:
        return float(str(x).replace(",", "."))
    except:
        return None


def snap_depth(d):
    return round(round(d / 1.5) * 1.5, 1)


def ocr_text(img):
    return pytesseract.image_to_string(img, lang="fra+eng", config="--psm 6")


def ocr_data(img):
    return pytesseract.image_to_data(
        img,
        lang="fra+eng",
        config="--psm 6",
        output_type=pytesseract.Output.DATAFRAME
    )


def keep_color_only(img, color):
    arr = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    if color == "blue":
        mask = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([145, 255, 255]))
    else:
        mask1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([15, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 40, 40]), np.array([180, 255, 255]))
        mask = mask1 + mask2

    result = np.ones_like(arr) * 255
    result[mask > 0] = arr[mask > 0]
    return Image.fromarray(result)


def clean_lithology(t):
    t = str(t).lower()
    t = t.replace("lerre", "terre")
    t = t.replace("tufcalcaire", "tuf calcaire")
    t = t.replace("caleaire", "calcaire")
    t = t.replace("calcaïre", "calcaire")
    t = t.replace("schiteuse", "schisteuse")
    t = t.replace("graniste", "granitique")
    t = re.sub(r"[^a-zA-Zéèêàùçîïôûâ\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    if "terre" in t:
        return "Terre végétale"
    if "tuf" in t and "calcaire" in t:
        return "Tuf calcaire"
    if "tuf" in t and "graveleux" in t:
        return "Tuf graveleux"
    if "calcaire" in t and ("dur" in t or "dure" in t):
        return "Calcaire dure"
    if "calcaire" in t:
        return "Calcaire"
    if "schiste" in t:
        return "Roche schisteuse dure"
    if "granit" in t:
        return "Roche granitique grise"
    if "argile" in t and "matrice" in t:
        return "Argile à matrice rocheuse"
    if "sable" in t and "matrice" in t:
        return "Sable à matrice rocheuse"
    if "argile" in t:
        return "Argile"
    if "sable" in t:
        return "Sable"
    if "marne" in t:
        return "Marne"
    return t.capitalize()


def extract_header(img):
    text = ocr_text(crop(img, HEADER_BOX))

    sondage = ""
    m = re.search(r"SP\s*[_\-]?\s*Reta\s*[_\-]?\s*(\d+)", text, re.I)
    if m:
        sondage = "SP_Reta" + m.group(1).zfill(3)
    else:
        m = re.search(r"(SP\s*[_\-]?\s*[A-Za-z]+\s*[_\-]?\s*\d+)", text, re.I)
        if m:
            sondage = re.sub(r"\s+", "", m.group(1)).replace("-", "_")

    coords = re.findall(r"\d{5,6}[.,]\d+", text)
    x = fr(coords[0]) if len(coords) >= 1 else ""
    y = fr(coords[1]) if len(coords) >= 2 else ""

    return sondage, x, y


def extract_lithologies(img):
    text = ocr_text(crop(img, LITHO_BOX))
    lines = [clean_lithology(x) for x in text.split("\n") if len(x.strip()) > 2]
    lines = [x for x in lines if x]

    if not lines:
        return []

    clean = []
    for l in lines:
        if l not in clean:
            clean.append(l)

    return clean


def lithology_for_depth(depth, lithos):
    if not lithos:
        return ""

    if len(lithos) == 1:
        return lithos[0]

    if depth <= 2.5:
        return lithos[0]

    return lithos[-1]


def extract_values(img, box, color, vmin, vmax):
    zone = crop(img, box)
    color_zone = keep_color_only(zone, color)

    df = ocr_data(color_zone)
    df = df.dropna(subset=["text"])

    h = color_zone.size[1]
    values = []

    for _, r in df.iterrows():
        txt = str(r["text"]).strip()
        txt = txt.replace("O", "0").replace("o", "0").replace("|", "1").replace("l", "1")

        nums = re.findall(r"\d+[.,]\d+|\d+", txt)

        for n in nums:
            val = to_float(n)
            if val is None:
                continue

            if vmin <= val <= vmax:
                cy = r["top"] + r["height"] / 2
                depth_raw = (cy / h) * DEPTH_MAX
                depth = snap_depth(depth_raw)

                if 0 < depth <= DEPTH_MAX:
                    values.append({
                        "depth": depth,
                        "value": val
                    })

    values = sorted(values, key=lambda x: x["depth"])

    clean = []
    seen = set()
    for v in values:
        if v["depth"] not in seen:
            clean.append(v)
            seen.add(v["depth"])

    return clean


def merge_pl_em(pls, ems):
    rows = []
    used = set()

    for pl in pls:
        best_i = None
        best_em = None
        best_dist = 999

        for i, em in enumerate(ems):
            if i in used:
                continue
            dist = abs(pl["depth"] - em["depth"])
            if dist < best_dist:
                best_dist = dist
                best_i = i
                best_em = em

        if best_em is not None and best_dist <= 0.8:
            used.add(best_i)
            rows.append({
                "depth": pl["depth"],
                "pl": pl["value"],
                "em": best_em["value"]
            })
        else:
            rows.append({
                "depth": pl["depth"],
                "pl": pl["value"],
                "em": ""
            })

    return rows


def process_page(img, page_num):
    sondage, x, y = extract_header(img)
    lithos = extract_lithologies(img)

    pls = extract_values(img, PL_BOX, "blue", 0.1, 30)
    ems = extract_values(img, EM_BOX, "red", 1, 20000)

    rows_data = merge_pl_em(pls, ems)

    rows = []
    for i, r in enumerate(rows_data):
        d = r["depth"]

        rows.append({
            "Page PDF": page_num,
            "Nom du sondages": sondage,
            "x": x if i == 0 else "",
            "y": y if i == 0 else "",
            "Profondeur (m)": fr(d),
            "Lithologie": lithology_for_depth(d, lithos),
            "Pl* (MPa)": fr(round(r["pl"], 3)),
            "Em (MPa)": fr(round(r["em"], 1)) if r["em"] != "" else ""
        })

    return rows


def make_excel(df):
    out = BytesIO()

    export = df[[
        "Nom du sondages",
        "x",
        "y",
        "Profondeur (m)",
        "Lithologie",
        "Pl* (MPa)",
        "Em (MPa)"
    ]]

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        export.to_excel(writer, index=False, startrow=1, sheet_name="Pressiometrique")
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

        for col, width in {"A": 18, "B": 15, "C": 15, "D": 15, "E": 35, "F": 14, "G": 14}.items():
            ws.column_dimensions[col].width = width

    out.seek(0)
    return out


if uploaded:
    if st.button("Extraire Excel"):
        with st.spinner("Extraction en cours..."):
            doc = fitz.open(stream=uploaded.read(), filetype="pdf")
            all_rows = []

            progress = st.progress(0)

            for i, page in enumerate(doc):
                pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))
                img = Image.open(BytesIO(pix.tobytes("png")))

                all_rows.extend(process_page(img, i + 1))
                progress.progress((i + 1) / len(doc))

            df = pd.DataFrame(all_rows)

        st.subheader("Résultat extrait")
        st.dataframe(df, use_container_width=True)

        excel = make_excel(df)

        st.download_button(
            "Télécharger Excel",
            data=excel,
            file_name="extraction_pressiometrique.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
