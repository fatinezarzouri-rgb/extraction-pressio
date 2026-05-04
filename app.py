import streamlit as st
import pandas as pd
import fitz, re, cv2, pytesseract, numpy as np
from PIL import Image
from io import BytesIO

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.title("Extraction pressiométrique PDF → Excel")
uploaded = st.file_uploader("Importer PDF", type=["pdf"])

DEPTHS = [1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12, 13.5, 15]

HEADER_BOX = (0.20, 0.08, 0.99, 0.30)
LITHO_BOX  = (0.15, 0.28, 0.42, 0.90)
PL_BOX     = (0.58, 0.28, 0.78, 0.90)
EM_BOX     = (0.78, 0.28, 0.98, 0.90)

def crop(img, box):
    w, h = img.size
    return img.crop((int(w*box[0]), int(h*box[1]), int(w*box[2]), int(h*box[3])))

def fr(x):
    return "" if x == "" else str(x).replace(".", ",")

def extract_header(img):
    text = pytesseract.image_to_string(crop(img, HEADER_BOX), lang="fra+eng", config="--psm 11")

    m = re.search(r"(SP\s*[_\-]?\s*[A-Za-z]+\s*[_\-]?\s*\d+)", text, re.I)
    sondage = re.sub(r"\s+", "", m.group(1)).replace("-", "_") if m else ""
    sondage = sondage.replace("_0", "0")

    coords = re.findall(r"\d{3}\s?\d{3}[.,]\d+", text)
    x = coords[0].replace(" ", "").replace(".", ",") if len(coords) > 0 else ""
    y = coords[1].replace(" ", "").replace(".", ",") if len(coords) > 1 else ""

    return sondage, x, y

def color_to_bw(img, color):
    arr = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    if color == "blue":
        mask = cv2.inRange(hsv, np.array([90, 30, 30]), np.array([150, 255, 255]))
    else:
        m1 = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([15, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([160, 30, 30]), np.array([180, 255, 255]))
        mask = m1 + m2

    out = np.ones(mask.shape, dtype=np.uint8) * 255
    out[mask > 0] = 0
    out = cv2.dilate(out, np.ones((2, 2), np.uint8), iterations=1)
    return Image.fromarray(out)

def fix_pl(v):
    # exemple OCR: 741 au lieu de 7.41
    if v > 30:
        return round(v / 100, 3)
    return v

def extract_column_values(img, box, color, vmin, vmax, is_pl=False):
    zone = crop(img, box)
    bw = color_to_bw(zone, color)

    text = pytesseract.image_to_string(
        bw,
        lang="eng",
        config="--psm 6 -c tessedit_char_whitelist=0123456789.,"
    )

    nums = re.findall(r"\d+[.,]\d+|\d+", text)
    vals = []

    for n in nums:
        try:
            v = float(n.replace(",", "."))
            if is_pl:
                v = fix_pl(v)
            if vmin <= v <= vmax:
                vals.append(v)
        except:
            pass

    return vals[:10]

def extract_litho(img):
    text = pytesseract.image_to_string(crop(img, LITHO_BOX), lang="fra+eng", config="--psm 6").lower()

    first = "Tuf calcaire" if "tuf" in text else ""

    if "calcaire" in text:
        second = "Calcaire dure"
    elif "schist" in text or "schiteuse" in text:
        second = "Roche schisteuse dure"
    elif "granit" in text or "graniste" in text:
        second = "Roche granitique grise"
    else:
        second = first

    return first, second

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
        litho1, litho2 = extract_litho(img)

        pl_vals = extract_column_values(img, PL_BOX, "blue", 0.1, 30, is_pl=True)
        em_vals = extract_column_values(img, EM_BOX, "red", 10, 20000, is_pl=False)

        for i, depth in enumerate(DEPTHS):
            rows.append({
                "Nom du sondages": sondage,
                "x": x if i == 0 else "",
                "y": y if i == 0 else "",
                "Profondeur (m)": fr(depth),
                "Lithologie": litho1 if depth <= 2.5 else litho2,
                "Pl* (MPa)": fr(round(pl_vals[i], 3)) if i < len(pl_vals) else "",
                "Em (MPa)": fr(round(em_vals[i], 1)) if i < len(em_vals) else ""
            })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Télécharger Excel",
        make_excel(df),
        "extraction_pressiometrique.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
