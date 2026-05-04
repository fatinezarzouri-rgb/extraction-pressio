import streamlit as st
import pandas as pd
import fitz, re, cv2, pytesseract, numpy as np
from PIL import Image
from io import BytesIO

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.set_page_config(page_title="Extraction Pressiométrique", layout="wide")
st.title("Extraction automatique PDF pressiométrique → Excel")

uploaded = st.file_uploader("Importer PDF", type=["pdf"])

DEPTH_MAX = 15.0

HEADER_BOX = (0.25, 0.11, 0.98, 0.26)
LITHO_BOX  = (0.15, 0.28, 0.42, 0.90)
PL_BOX     = (0.62, 0.28, 0.78, 0.90)   # bleu
EM_BOX     = (0.81, 0.28, 0.97, 0.90)   # rouge à droite seulement


def crop(img, box):
    w, h = img.size
    return img.crop((int(w*box[0]), int(h*box[1]), int(w*box[2]), int(h*box[3])))


def fr(x):
    if x == "" or x is None:
        return ""
    return str(x).replace(".", ",")


def snap_depth(d):
    return round(round(d / 1.5) * 1.5, 1)


def clean_num(v):
    return float(str(v).replace(",", "."))


def keep_color(img, color):
    arr = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    if color == "blue":
        mask = cv2.inRange(hsv, np.array([90, 35, 35]), np.array([145, 255, 255]))
    else:
        m1 = cv2.inRange(hsv, np.array([0, 35, 35]), np.array([15, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([160, 35, 35]), np.array([180, 255, 255]))
        mask = m1 + m2

    result = np.ones_like(arr) * 255
    result[mask > 0] = arr[mask > 0]
    return Image.fromarray(result)


def ocr_text(img):
    return pytesseract.image_to_string(img, lang="fra+eng", config="--psm 6")


def extract_header(img):
    text = ocr_text(crop(img, HEADER_BOX))

    sondage = ""
    m = re.search(r"(SP\s*[_\-]?\s*[A-Za-z]+\s*[_\-]?\s*\d+)", text, re.I)
    if m:
        sondage = re.sub(r"\s+", "", m.group(1)).replace("-", "_")

    coords = re.findall(r"\d{5,6}[.,]\d+", text)
    x = fr(coords[0]) if len(coords) >= 1 else ""
    y = fr(coords[1]) if len(coords) >= 2 else ""

    return sondage, x, y


def clean_litho(txt):
    t = txt.lower()
    t = t.replace("lerre", "terre").replace("tufcalcaire", "tuf calcaire")
    t = t.replace("caleaire", "calcaire").replace("calcaïre", "calcaire")
    t = t.replace("schiteuse", "schisteuse").replace("graniste", "granitique")

    if "terre" in t:
        return "Terre végétale"
    if "tuf" in t and "calcaire" in t:
        return "Tuf calcaire"
    if "tuf" in t and "graveleux" in t:
        return "Tuf graveleux"
    if "calcaire" in t:
        return "Calcaire dure"
    if "schiste" in t:
        return "Roche schisteuse dure"
    if "granit" in t:
        return "Roche granitique grise"
    if "argile" in t:
        return "Argile"
    if "sable" in t:
        return "Sable"
    if "marne" in t:
        return "Marne"
    return ""


def extract_lithologies(img):
    text = ocr_text(crop(img, LITHO_BOX))
    lines = [clean_litho(l) for l in text.split("\n")]
    lines = [l for l in lines if l]

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
    zone = keep_color(zone, color)

    df = pytesseract.image_to_data(
        zone,
        lang="eng",
        config="--psm 6",
        output_type=pytesseract.Output.DATAFRAME
    )

    df = df.dropna(subset=["text"])
    h = zone.size[1]
    values = []

    for _, r in df.iterrows():
        txt = str(r["text"]).strip()
        txt = txt.replace("O", "0").replace("o", "0").replace("|", "1").replace("l", "1")

        nums = re.findall(r"\d+[.,]\d+|\d+", txt)

        for n in nums:
            try:
                val = clean_num(n)
            except:
                continue

            if vmin <= val <= vmax:
                cy = r["top"] + r["height"] / 2
                depth = snap_depth((cy / h) * DEPTH_MAX)

                if 0 < depth <= DEPTH_MAX:
                    values.append((depth, val))

    values = sorted(values, key=lambda x: x[0])

    final = {}
    for d, v in values:
        final[d] = v

    return sorted(final.items())


def merge_values(pls, ems):
    rows = []
    em_dict = dict(ems)

    for d, pl in pls:
        em = em_dict.get(d, "")
        rows.append((d, pl, em))

    return rows


def make_excel(df):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export = df[[
            "Nom du sondages", "x", "y", "Profondeur (m)",
            "Lithologie", "Pl* (MPa)", "Em (MPa)"
        ]]

        export.to_excel(writer, index=False, startrow=1, sheet_name="Pressiometrique")
        ws = writer.book["Pressiometrique"]

        ws.merge_cells("A1:D1")
        ws["A1"] = "Echantillon"
        ws.merge_cells("E1:G1")
        ws["E1"] = "Caracteristiques pressiometriques"

    output.seek(0)
    return output


if uploaded:
    if st.button("Extraire Excel"):
        doc = fitz.open(stream=uploaded.read(), filetype="pdf")
        rows = []

        progress = st.progress(0)

        for page_index, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))
            img = Image.open(BytesIO(pix.tobytes("png")))

            sondage, x, y = extract_header(img)
            lithos = extract_lithologies(img)

            pls = extract_values(img, PL_BOX, "blue", 0.1, 30)
            ems = extract_values(img, EM_BOX, "red", 10, 20000)

            merged = merge_values(pls, ems)

            for i, (depth, pl, em) in enumerate(merged):
                rows.append({
                    "Nom du sondages": sondage,
                    "x": x if i == 0 else "",
                    "y": y if i == 0 else "",
                    "Profondeur (m)": fr(depth),
                    "Lithologie": lithology_for_depth(depth, lithos),
                    "Pl* (MPa)": fr(round(pl, 3)),
                    "Em (MPa)": fr(round(em, 1)) if em != "" else ""
                })

            progress.progress((page_index + 1) / len(doc))

        df = pd.DataFrame(rows)

        st.dataframe(df, use_container_width=True)

        excel = make_excel(df)

        st.download_button(
            "Télécharger Excel",
            excel,
            "extraction_pressiometrique.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
