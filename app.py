import streamlit as st
import pandas as pd
import fitz
import pytesseract
import re
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.set_page_config(page_title="Extraction pressiométrique", layout="wide")
st.title("Extraction PDF pressiométrique vers Excel")

uploaded = st.file_uploader("Importer le PDF", type=["pdf"])

DEPTH_MAX = 15.0

HEADER_BOX = (0.30, 0.12, 0.98, 0.25)
LITHO_BOX  = (0.12, 0.28, 0.42, 0.90)
PL_BOX     = (0.55, 0.28, 0.80, 0.90)
EM_BOX     = (0.75, 0.28, 0.98, 0.90)


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


def keep_color_only(img, color):
    arr = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    if color == "blue":
        lower = np.array([90, 40, 40])
        upper = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

    elif color == "red":
        lower1 = np.array([0, 40, 40])
        upper1 = np.array([15, 255, 255])
        lower2 = np.array([160, 40, 40])
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = mask1 + mask2

    else:
        mask = np.zeros(arr.shape[:2], dtype=np.uint8)

    result = np.ones_like(arr) * 255
    result[mask > 0] = arr[mask > 0]

    return Image.fromarray(result)


def ocr_text(img):
    return pytesseract.image_to_string(img, lang="fra+eng", config="--psm 6")


def ocr_data(img):
    return pytesseract.image_to_data(
        img,
        lang="fra+eng",
        config="--psm 6",
        output_type=pytesseract.Output.DATAFRAME
    )


def extract_header(img):
    zone = crop(img, HEADER_BOX)
    text = ocr_text(zone)

    sondage = ""
    x = ""
    y = ""

    m = re.search(r"(SP[_\-]?[A-Za-z]+[_\-]?\d+)", text, re.I)
    if m:
        sondage = m.group(1).replace("-", "_")

    mx = re.search(r"X\s*[:\-]?\s*(\d+[.,]\d+)", text)
    my = re.search(r"Y\s*[:\-]?\s*(\d+[.,]\d+)", text)

    if mx:
        x = fr(mx.group(1))
    if my:
        y = fr(my.group(1))

    return sondage, x, y


def extract_values_by_color(zone_img, color, vmin, vmax):
    color_img = keep_color_only(zone_img, color)
    df = ocr_data(color_img)
    df = df.dropna(subset=["text"])

    h = color_img.size[1]
    values = []

    for _, r in df.iterrows():
        txt = str(r["text"]).strip()
        txt = txt.replace("O", "0").replace("o", "0")
        txt = txt.replace("|", "1").replace("l", "1")

        found = re.findall(r"\d+[.,]\d+|\d+", txt)

        for item in found:
            val = to_float(item)
            if val is None:
                continue

            if vmin <= val <= vmax:
                cy = r["top"] + r["height"] / 2
                depth = round((cy / h) * DEPTH_MAX, 2)

                values.append({
                    "depth": depth,
                    "value": val
                })

    values = sorted(values, key=lambda x: x["depth"])

    clean = []
    for v in values:
        if not clean or abs(v["depth"] - clean[-1]["depth"]) > 0.25:
            clean.append(v)

    return clean


def extract_lithologies(zone_img):
    text = ocr_text(zone_img).lower()

    lithos = []

    known = [
        "terre végétale",
        "tuf calcaire",
        "tuf graveleux",
        "calcaire dure",
        "calcaire dur",
        "roche schisteuse dure",
        "roche schiteuse dure",
        "roche granitique grise",
        "roche graniste grise",
        "argile à matrice rocheuse",
        "sable à matrice rocheuse",
        "argile",
        "limon",
        "sable",
        "marne",
        "grès",
        "schiste"
    ]

    for k in known:
        if k in text:
            lithos.append(k.capitalize())

    if not lithos:
        lithos = [""]

    if len(lithos) == 1:
        return [{"z1": 0, "z2": DEPTH_MAX, "lithologie": lithos[0]}]

    if len(lithos) == 2:
        return [
            {"z1": 0, "z2": 2.5, "lithologie": lithos[0]},
            {"z1": 2.5, "z2": DEPTH_MAX, "lithologie": lithos[1]}
        ]

    return [
        {"z1": 0, "z2": 0.5, "lithologie": lithos[0]},
        {"z1": 0.5, "z2": 2.5, "lithologie": lithos[1]},
        {"z1": 2.5, "z2": DEPTH_MAX, "lithologie": lithos[-1]}
    ]


def lithology_at_depth(depth, intervals):
    for c in intervals:
        if c["z1"] <= depth <= c["z2"]:
            return c["lithologie"]
    return intervals[-1]["lithologie"] if intervals else ""


def merge_pl_em(pl_values, em_values):
    rows = []
    used_em = set()

    for pl in pl_values:
        best_i = None
        best_em = None
        best_dist = 999

        for i, em in enumerate(em_values):
            if i in used_em:
                continue

            dist = abs(pl["depth"] - em["depth"])
            if dist < best_dist:
                best_dist = dist
                best_i = i
                best_em = em

        if best_em and best_dist <= 0.8:
            used_em.add(best_i)
            depth = round((pl["depth"] + best_em["depth"]) / 2, 2)
            rows.append({
                "depth": depth,
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

    litho_zone = crop(img, LITHO_BOX)
    pl_zone = crop(img, PL_BOX)
    em_zone = crop(img, EM_BOX)

    lithos = extract_lithologies(litho_zone)

    pl_values = extract_values_by_color(pl_zone, "blue", 0.1, 20)
    em_values = extract_values_by_color(em_zone, "red", 1, 10000)

    merged = merge_pl_em(pl_values, em_values)

    rows = []

    for i, r in enumerate(merged):
        depth = r["depth"]

        rows.append({
            "Page PDF": page_num,
            "Nom du sondages": sondage,
            "x": x if i == 0 else "",
            "y": y if i == 0 else "",
            "Profondeur (m)": fr(round(depth, 2)),
            "Lithologie": lithology_at_depth(depth, lithos),
            "Pl* (MPa)": fr(round(r["pl"], 3)),
            "Em (MPa)": fr(round(r["em"], 1)) if r["em"] != "" else ""
        })

    return rows


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
            "A": 18,
            "B": 15,
            "C": 15,
            "D": 15,
            "E": 35,
            "F": 14,
            "G": 14
        }

        for col, width in widths.items():
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

                rows = process_page(img, i + 1)
                all_rows.extend(rows)

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
