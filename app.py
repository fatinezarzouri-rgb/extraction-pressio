import streamlit as st
import pandas as pd
import fitz, re, pytesseract
from PIL import Image
from io import BytesIO

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.title("Extraction PDF pressiométrique → Excel")

uploaded = st.file_uploader("Importer PDF", type=["pdf"])

HEADER_BOX = (0.20, 0.08, 0.99, 0.30)
LITHO_BOX  = (0.15, 0.28, 0.42, 0.90)
TABLE_BOX  = (0.45, 0.28, 0.99, 0.90)

DEPTHS = [1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12, 13.5, 15]


def crop(img, box):
    w, h = img.size
    return img.crop((int(w*box[0]), int(h*box[1]), int(w*box[2]), int(h*box[3])))


def fr(x):
    return str(x).replace(".", ",")


def to_float(x):
    try:
        return float(str(x).replace(",", "."))
    except:
        return None


def ocr(img, psm=6):
    return pytesseract.image_to_string(img, lang="fra+eng", config=f"--psm {psm}")


def extract_header(img):
    text = ocr(crop(img, HEADER_BOX))

    m = re.search(r"(SP\s*[_\-]?\s*[A-Za-z]+\s*[_\-]?\s*\d+)", text, re.I)
    sondage = ""
    if m:
        sondage = re.sub(r"\s+", "", m.group(1)).replace("-", "_")
        sondage = re.sub(r"_+(\d+)$", r"\1", sondage)  # SP_Reta_043 -> SP_Reta043

    coords = re.findall(r"\d{3}\s?\d{3}[.,]\d+", text)
    x = coords[0].replace(" ", "").replace(".", ",") if len(coords) > 0 else ""
    y = coords[1].replace(" ", "").replace(".", ",") if len(coords) > 1 else ""

    return sondage, x, y


def lithologies(img):
    text = ocr(crop(img, LITHO_BOX)).lower()

    first = "Tuf calcaire" if "tuf" in text else ""
    if "calcaire" in text:
        second = "Calcaire dure"
    elif "schiste" in text or "schiteuse" in text:
        second = "Roche schisteuse dure"
    elif "granit" in text or "graniste" in text:
        second = "Roche granitique grise"
    else:
        second = ""

    return first, second


def read_row_values(table_img, depth):
    w, h = table_img.size

    y = int((depth / 15) * h)
    band = table_img.crop((0, max(0, y-35), w, min(h, y+35)))

    text = ocr(band, psm=6)
    nums = re.findall(r"\d+[.,]\d+", text)
    vals = [to_float(n) for n in nums]
    vals = [v for v in vals if v is not None]

    # Ligne type : Pf / Pl / Em
    candidates = []
    for i in range(len(vals)-2):
        pf, pl, em = vals[i], vals[i+1], vals[i+2]
        if 0.1 <= pf <= 20 and 0.1 <= pl <= 30 and 10 <= em <= 20000:
            candidates.append((pl, em))

    if candidates:
        return candidates[-1]

    return "", ""


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
        litho1, litho2 = lithologies(img)
        table_img = crop(img, TABLE_BOX)

        for i, d in enumerate(DEPTHS):
            pl, em = read_row_values(table_img, d)

            rows.append({
                "Nom du sondages": sondage,
                "x": x if i == 0 else "",
                "y": y if i == 0 else "",
                "Profondeur (m)": fr(d),
                "Lithologie": litho1 if d <= 2.5 else litho2,
                "Pl* (MPa)": fr(pl) if pl != "" else "",
                "Em (MPa)": fr(em) if em != "" else ""
            })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Télécharger Excel",
        make_excel(df),
        "extraction_pressiometrique.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
