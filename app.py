import streamlit as st
import pandas as pd
import fitz
import pytesseract
import cv2
import numpy as np
import re
from PIL import Image
from io import BytesIO

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.title("Extraction PDF pressiométrique → Excel")

uploaded = st.file_uploader("Importer PDF", type=["pdf"])

DEPTHS = [1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12, 13.5, 15]

HEADER_BOX = (0.20, 0.08, 0.99, 0.30)
LITHO_BOX  = (0.15, 0.28, 0.42, 0.90)
PL_BOX     = (0.62, 0.28, 0.78, 0.90)
EM_BOX     = (0.79, 0.28, 0.98, 0.90)


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


def ocr(img, psm=6):
    return pytesseract.image_to_string(img, lang="fra+eng", config=f"--psm {psm}")


def color_only(img, color):
    arr = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    if color == "blue":
        mask = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([145, 255, 255]))
    else:
        m1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([15, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([160, 40, 40]), np.array([180, 255, 255]))
        mask = m1 + m2

    out = np.ones(mask.shape, dtype=np.uint8) * 255
    out[mask > 0] = 0
    out = cv2.dilate(out, np.ones((2, 2), np.uint8), iterations=1)

    return Image.fromarray(out)


def extract_header(img):
    text = ocr(crop(img, HEADER_BOX), 6)

    m = re.search(r"(SP\s*[_\-]?\s*[A-Za-z]+\s*[_\-]?\s*\d+)", text, re.I)
    sondage = ""
    if m:
        sondage = re.sub(r"\s+", "", m.group(1)).replace("-", "_")
        sondage = sondage.replace("_0", "0")

    coords = re.findall(r"\d{3}\s?\d{3}[.,]\d+", text)
    x = coords[0].replace(" ", "").replace(".", ",") if len(coords) > 0 else ""
    y = coords[1].replace(" ", "").replace(".", ",") if len(coords) > 1 else ""

    return sondage, x, y


def extract_lithology(img):
    text = ocr(crop(img, LITHO_BOX), 6).lower()

    if "tuf" in text and "calcaire" in text:
        litho1 = "Tuf calcaire"
    elif "tuf" in text and "graveleux" in text:
        litho1 = "Tuf graveleux"
    else:
        litho1 = ""

    if "calcaire" in text:
        litho2 = "Calcaire dure"
    elif "schiste" in text or "schiteuse" in text:
        litho2 = "Roche schisteuse dure"
    elif "granit" in text or "graniste" in text:
        litho2 = "Roche granitique grise"
    else:
        litho2 = litho1

    return litho1, litho2


def extract_pl_points(img):
    pl_zone = crop(img, PL_BOX)
    bw = color_only(pl_zone, "blue")

    df = pytesseract.image_to_data(
        bw,
        lang="eng",
        config="--psm 6 -c tessedit_char_whitelist=0123456789.,",
        output_type=pytesseract.Output.DATAFRAME
    )

    df = df.dropna(subset=["text"])

    h = pl_zone.size[1]
    values = []

    page_w, page_h = img.size
    pl_y0_global = int(page_h * PL_BOX[1])

    for _, r in df.iterrows():
        txt = str(r["text"]).strip()
        nums = re.findall(r"\d+[.,]\d+|\d+", txt)

        for n in nums:
            v = to_float(n)
            if v is None:
                continue

            if 0.1 <= v <= 30:
                y_local = r["top"] + r["height"] / 2
                y_global = pl_y0_global + y_local
                depth = snap_depth((y_local / h) * 15)

                values.append({
                    "depth": depth,
                    "pl": v,
                    "y_global": y_global
                })

    values = sorted(values, key=lambda a: a["depth"])

    clean = {}
    for v in values:
        clean[v["depth"]] = v

    return [clean[d] for d in sorted(clean.keys())]


def get_em_for_pl(img, y_global):
    page_w, page_h = img.size

    em_zone = crop(img, EM_BOX)
    em_y0_global = int(page_h * EM_BOX[1])

    y_local = int(y_global - em_y0_global)

    w, h = em_zone.size

    band = em_zone.crop((
        0,
        max(0, y_local - 35),
        w,
        min(h, y_local + 35)
    ))

    bw = color_only(band, "red")

    text = pytesseract.image_to_string(
        bw,
        lang="eng",
        config="--psm 7 -c tessedit_char_whitelist=0123456789.,"
    )

    nums = re.findall(r"\d+[.,]\d+|\d+", text)

    vals = []
    for n in nums:
        v = to_float(n)
        if v is not None and 10 <= v <= 20000:
            vals.append(v)

    if vals:
        return vals[-1]

    return ""


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


if uploaded and st.button("Extraire Excel"):
    doc = fitz.open(stream=uploaded.read(), filetype="pdf")
    rows = []

    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))
        img = Image.open(BytesIO(pix.tobytes("png")))

        sondage, x, y = extract_header(img)
        litho1, litho2 = extract_lithology(img)

        pl_points = extract_pl_points(img)

        for i, p in enumerate(pl_points):
            depth = p["depth"]
            em = get_em_for_pl(img, p["y_global"])

            rows.append({
                "Nom du sondages": sondage,
                "x": x if i == 0 else "",
                "y": y if i == 0 else "",
                "Profondeur (m)": fr(depth),
                "Lithologie": litho1 if depth <= 2.5 else litho2,
                "Pl* (MPa)": fr(round(p["pl"], 3)),
                "Em (MPa)": fr(round(em, 1)) if em != "" else ""
            })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Télécharger Excel",
        make_excel(df),
        "extraction_pressiometrique.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
