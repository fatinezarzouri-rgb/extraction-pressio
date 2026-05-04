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

HEADER_BOX = (0.28, 0.12, 0.99, 0.26)
LOG_BOX = (0.07, 0.28, 0.39, 0.90)
LITHO_BOX = (0.16, 0.28, 0.40, 0.90)
PL_BOX = (0.63, 0.28, 0.78, 0.90)
EM_BOX = (0.80, 0.28, 0.97, 0.90)


def crop(img, box):
    w, h = img.size
    return img.crop((int(w*box[0]), int(h*box[1]), int(w*box[2]), int(h*box[3])))


def fr(x):
    if x is None or x == "":
        return ""
    return str(x).replace(".", ",")


def to_float(x):
    try:
        return float(str(x).replace(",", "."))
    except:
        return None


def ocr_text(img):
    return pytesseract.image_to_string(img, lang="fra+eng", config="--psm 6")


def ocr_data(img):
    return pytesseract.image_to_data(
        img,
        lang="fra+eng",
        config="--psm 6",
        output_type=pytesseract.Output.DATAFRAME
    )


def clean_lithology(text):
    text = str(text).lower()

    text = text.replace("lerre", "terre")
    text = text.replace("ierre", "terre")
    text = text.replace("vegetale", "végétale")
    text = text.replace("tufcalcaire", "tuf calcaire")
    text = text.replace("caleaire", "calcaire")
    text = text.replace("calcaïre", "calcaire")
    text = text.replace("dure cr", "dure")
    text = text.replace("dur cr", "dur")
    text = text.replace("schiteuse", "schisteuse")
    text = text.replace("graniste", "granitique")

    text = re.sub(r"[^a-zA-Zéèêàùçîïôûâ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if "terre" in text:
        return "Terre végétale"
    if "tuf" in text and "calcaire" in text:
        return "Tuf calcaire"
    if "tuf" in text and "graveleux" in text:
        return "Tuf graveleux"
    if "calcaire" in text and ("dur" in text or "dure" in text):
        return "Calcaire dure"
    if "calcaire" in text:
        return "Calcaire"
    if "schiste" in text:
        return "Roche schisteuse dure"
    if "granit" in text:
        return "Roche granitique grise"
    if "argile" in text and "matrice" in text:
        return "Argile à matrice rocheuse"
    if "sable" in text and "matrice" in text:
        return "Sable à matrice rocheuse"
    if "argile" in text:
        return "Argile"
    if "sable" in text:
        return "Sable"
    if "marne" in text:
        return "Marne"
    if "gres" in text or "grès" in text:
        return "Grès"

    return text.capitalize()


def extract_header(img):
    zone = crop(img, HEADER_BOX)
    text = ocr_text(zone)

    sondage = ""
    x = ""
    y = ""

    m = re.search(r"SP\s*[_\-]?\s*Reta\s*[_\-]?\s*(\d+)", text, re.I)
    if m:
        sondage = "SP_Reta" + m.group(1).zfill(3)
    else:
        m2 = re.search(r"(SP\s*[_\-]?\s*[A-Za-z]+\s*[_\-]?\s*\d+)", text, re.I)
        if m2:
            sondage = re.sub(r"\s+", "", m2.group(1)).replace("-", "_")

    coords = re.findall(r"\d{5,6}[.,]\d+", text)

    if len(coords) >= 2:
        x = fr(coords[0])
        y = fr(coords[1])

    return sondage, x, y


def keep_color_only(img, color):
    arr = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    if color == "blue":
        lower = np.array([90, 35, 35])
        upper = np.array([145, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    else:
        lower1 = np.array([0, 35, 35])
        upper1 = np.array([15, 255, 255])
        lower2 = np.array([160, 35, 35])
        upper2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower1, upper1) + cv2.inRange(hsv, lower2, upper2)

    result = np.ones_like(arr) * 255
    result[mask > 0] = arr[mask > 0]
    return Image.fromarray(result)


def get_axis_bounds(log_img):
    return 0, log_img.size[1]


def y_to_depth(y, top, bottom):
    return ((y - top) / (bottom - top)) * DEPTH_MAX


def detect_layer_lines(log_img, top, bottom):
    arr = np.array(log_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=70,
        minLineLength=int(arr.shape[1] * 0.25),
        maxLineGap=8
    )

    ys = [top, bottom]

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) <= 4:
                y = int((y1 + y2) / 2)
                if top <= y <= bottom:
                    ys.append(y)

    ys = sorted(ys)

    clean = []
    for y in ys:
        if not clean or abs(y - clean[-1]) > 25:
            clean.append(y)

    return clean


def extract_lithology_texts(litho_img, top, bottom):
    df = ocr_data(litho_img)
    df = df.dropna(subset=["text"])

    words = []

    for _, r in df.iterrows():
        txt = str(r["text"]).strip()
        if len(txt) < 2:
            continue
        if not re.search(r"[A-Za-zéèêàùçîïôûâ]", txt):
            continue
        if txt.lower() in ["lithologie", "profondeur", "labotest", "m"]:
            continue

        cy = r["top"] + r["height"] / 2
        depth = y_to_depth(cy, top, bottom)

        if 0 <= depth <= DEPTH_MAX:
            words.append((depth, txt))

    words = sorted(words, key=lambda x: x[0])

    groups = []
    for depth, word in words:
        if not groups or abs(depth - groups[-1]["depth"]) > 0.8:
            groups.append({"depth": depth, "words": [word]})
        else:
            groups[-1]["words"].append(word)
            groups[-1]["depth"] = (groups[-1]["depth"] + depth) / 2

    return [{"depth": g["depth"], "text": clean_lithology(" ".join(g["words"]))} for g in groups]


def build_layers(log_img, litho_img):
    top, bottom = get_axis_bounds(log_img)
    lines = detect_layer_lines(log_img, top, bottom)
    litho_texts = extract_lithology_texts(litho_img, top, bottom)

    layers = []

    for i in range(len(lines) - 1):
        y1 = lines[i]
        y2 = lines[i + 1]

        z1 = round(y_to_depth(y1, top, bottom), 2)
        z2 = round(y_to_depth(y2, top, bottom), 2)
        mid = (z1 + z2) / 2

        litho = ""
        if litho_texts:
            nearest = min(litho_texts, key=lambda t: abs(t["depth"] - mid))
            litho = nearest["text"]

        if z2 > z1:
            layers.append({
                "z_debut": z1,
                "z_fin": z2,
                "lithologie": litho
            })

    return layers, top, bottom


def lithology_at_depth(depth, layers):
    for c in layers:
        if c["z_debut"] <= depth <= c["z_fin"]:
            return c["lithologie"]
    return ""


def extract_colored_values(zone_img, zone_box, color, vmin, vmax, page_img, axis_top_global, axis_bottom_global):
    color_img = keep_color_only(zone_img, color)

    df = ocr_data(color_img)
    df = df.dropna(subset=["text"])

    page_w, page_h = page_img.size
    zone_y1 = int(page_h * zone_box[1])

    values = []

    for _, r in df.iterrows():
        txt = str(r["text"]).strip()
        txt = txt.replace("O", "0").replace("o", "0")
        txt = txt.replace("|", "1").replace("l", "1")

        nums = re.findall(r"\d+[.,]\d+|\d+", txt)

        for n in nums:
            val = to_float(n)
            if val is None:
                continue

            if vmin <= val <= vmax:
                cy_global = zone_y1 + r["top"] + r["height"] / 2
                depth = y_to_depth(cy_global, axis_top_global, axis_bottom_global)

                if 0 <= depth <= DEPTH_MAX:
                    values.append({
                        "depth": round(depth, 2),
                        "value": val
                    })

    values = sorted(values, key=lambda x: x["depth"])

    clean = []
    for v in values:
        if not clean or abs(v["depth"] - clean[-1]["depth"]) > 0.20:
            clean.append(v)

    return clean


def merge_pl_em(pl_values, em_values):
    rows = []
    used = set()

    for pl in pl_values:
        best_i = None
        best_em = None
        best_dist = 999

        for i, em in enumerate(em_values):
            if i in used:
                continue
            dist = abs(pl["depth"] - em["depth"])
            if dist < best_dist:
                best_dist = dist
                best_i = i
                best_em = em

        if best_em is not None and best_dist <= 0.8:
            used.add(best_i)
            depth = round((pl["depth"] + best_em["depth"]) / 2, 2)
            rows.append({"depth": depth, "pl": pl["value"], "em": best_em["value"]})
        else:
            rows.append({"depth": pl["depth"], "pl": pl["value"], "em": ""})

    return rows


def process_page(img, page_num):
    sondage, x, y = extract_header(img)

    log_img = crop(img, LOG_BOX)
    litho_img = crop(img, LITHO_BOX)
    pl_img = crop(img, PL_BOX)
    em_img = crop(img, EM_BOX)

    layers, axis_top_local, axis_bottom_local = build_layers(log_img, litho_img)

    page_w, page_h = img.size
    log_y1_global = int(page_h * LOG_BOX[1])

    axis_top_global = log_y1_global + axis_top_local
    axis_bottom_global = log_y1_global + axis_bottom_local

    pl_values = extract_colored_values(
        pl_img, PL_BOX, "blue", 0.1, 30,
        img, axis_top_global, axis_bottom_global
    )

    em_values = extract_colored_values(
        em_img, EM_BOX, "red", 10, 20000,
        img, axis_top_global, axis_bottom_global
    )

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
            "Lithologie": lithology_at_depth(depth, layers),
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
