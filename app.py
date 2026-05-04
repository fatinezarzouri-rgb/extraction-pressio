import streamlit as st
import pandas as pd
import fitz, re, pytesseract
from PIL import Image
from io import BytesIO

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.title("Extraction pressiométrique PDF → Excel")
uploaded = st.file_uploader("Importer PDF", type=["pdf"])

DEPTHS = [1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12, 13.5, 15]

HEADER_BOX = (0.18, 0.08, 0.99, 0.30)
LITHO_BOX  = (0.14, 0.28, 0.42, 0.90)
PL_BOX     = (0.61, 0.28, 0.78, 0.90)
EM_BOX     = (0.80, 0.28, 0.98, 0.90)

def crop(img, box):
    w, h = img.size
    return img.crop((int(w*box[0]), int(h*box[1]), int(w*box[2]), int(h*box[3])))

def fr(x):
    return "" if x == "" or x is None else str(x).replace(".", ",")

def to_float(x):
    try:
        return float(str(x).replace(",", "."))
    except:
        return None

def ocr(img, psm=6):
    return pytesseract.image_to_string(
        img,
        lang="fra+eng",
        config=f"--psm {psm} -c tessedit_char_whitelist=0123456789.,ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_:- "
    )

def extract_header(img):
    text = ocr(crop(img, HEADER_BOX), 6)

    sondage = ""
    m = re.search(r"(SP\s*[_\-]?\s*[A-Za-z]+\s*[_\-]?\s*\d+)", text, re.I)
    if m:
        sondage = re.sub(r"\s+", "", m.group(1)).replace("-", "_").replace("_0", "0")

    coords = re.findall(r"\d{3}\s?\d{3}[.,]\d+", text)
    x = coords[0].replace(" ", "").replace(".", ",") if len(coords) > 0 else ""
    y = coords[1].replace(" ", "").replace(".", ",") if len(coords) > 1 else ""

    return sondage, x, y

def extract_lithology(img):
    text = ocr(crop(img, LITHO_BOX), 6).lower()

    if "tuf" in text and "graveleux" in text:
        litho1 = "Tuf graveleux"
    elif "tuf" in text:
        litho1 = "Tuf calcaire"
    else:
        litho1 = ""

    if "calcaire" in text:
        litho2 = "Calcaire dure"
    elif "schist" in text or "schiteuse" in text:
        litho2 = "Roche schisteuse dure"
    elif "granit" in text or "graniste" in text:
        litho2 = "Roche granitique grise"
    else:
        litho2 = litho1

    return litho1, litho2

def read_value_at_depth(img, box, depth, vmin, vmax):
    zone = crop(img, box)
    w, h = zone.size

    y = int((depth / 15) * h)
    band = zone.crop((0, max(0, y-32), w, min(h, y+32)))

    text = pytesseract.image_to_string(
        band,
        lang="eng",
        config="--psm 7 -c tessedit_char_whitelist=0123456789.,"
    )

    nums = re.findall(r"\d+[.,]\d+|\d+", text)
    vals = []

    for n in nums:
        v = to_float(n)
        if v is not None and vmin <= v <= vmax:
            vals.append(v)

    if not vals:
        return ""

    return max(vals)

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
        pix = page.get_pixmap(matrix=fitz.Matrix(5, 5))
        img = Image.open(BytesIO(pix.tobytes("png")))

        sondage, x, y = extract_header(img)
        litho1, litho2 = extract_lithology(img)

        for i, depth in enumerate(DEPTHS):
            pl = read_value_at_depth(img, PL_BOX, depth, 0.1, 30)
            em = read_value_at_depth(img, EM_BOX, depth, 10, 20000)

            rows.append({
                "Nom du sondages": sondage,
                "x": x if i == 0 else "",
                "y": y if i == 0 else "",
                "Profondeur (m)": fr(depth),
                "Lithologie": litho1 if depth <= 2.5 else litho2,
                "Pl* (MPa)": fr(round(pl, 3)) if pl != "" else "",
                "Em (MPa)": fr(round(em, 1)) if em != "" else ""
            })

    df = pd.DataFrame(rows)
    edited = st.data_editor(df, use_container_width=True, num_rows="dynamic")

    st.download_button(
        "Télécharger Excel",
        make_excel(edited),
        "extraction_pressiometrique.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
