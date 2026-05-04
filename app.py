import streamlit as st
import pandas as pd
import fitz
import pytesseract
import re
from PIL import Image
from io import BytesIO
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.set_page_config(page_title="Extraction pressiométrique", layout="wide")
st.title("Extraction PDF pressiométrique → Excel")

uploaded = st.file_uploader("Importer le PDF", type=["pdf"])

HEADER_BOX = (0.20, 0.08, 0.99, 0.30)
DATA_BOX = (0.05, 0.25, 0.99, 0.92)


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


def clean_coord(x):
    return x.replace(" ", "").replace(".", ",")


def ocr_text(img):
    return pytesseract.image_to_string(img, lang="fra+eng", config="--psm 6")


def extract_header(img):
    text = ocr_text(crop(img, HEADER_BOX))

    sondage = ""
    m = re.search(r"(SP\s*[_\-]?\s*[A-Za-z]+\s*[_\-]?\s*\d+)", text, re.I)
    if m:
        sondage = re.sub(r"\s+", "", m.group(1)).replace("-", "_")

    coords = re.findall(r"\d{3}\s?\d{3}[.,]\d+", text)
    x = clean_coord(coords[0]) if len(coords) >= 1 else ""
    y = clean_coord(coords[1]) if len(coords) >= 2 else ""

    return sondage, x, y


def clean_litho(text):
    t = text.lower()

    if "tuf" in t and "calcaire" in t:
        first = "Tuf calcaire"
    elif "tuf" in t and "graveleux" in t:
        first = "Tuf graveleux"
    else:
        first = ""

    if "calcaire" in t and ("dur" in t or "dure" in t):
        second = "Calcaire dure"
    elif "schiste" in t or "schiteuse" in t:
        second = "Roche schisteuse dure"
    elif "granit" in t or "graniste" in t:
        second = "Roche granitique grise"
    else:
        second = first

    return first, second


def extract_rows(img):
    zone = crop(img, DATA_BOX)

    df = pytesseract.image_to_data(
        zone,
        lang="fra+eng",
        config="--psm 6",
        output_type=pytesseract.Output.DATAFRAME
    )

    df = df.dropna(subset=["text"])
    h = zone.size[1]

    words = []

    for _, r in df.iterrows():
        txt = str(r["text"]).strip()
        if not txt:
            continue

        nums = re.findall(r"\d+[.,]\d+", txt)
        for n in nums:
            val = to_float(n)
            if val is None:
                continue

            cy = r["top"] + r["height"] / 2
            cx = r["left"] + r["width"] / 2

            words.append({
                "x": cx,
                "y": cy,
                "value": val
            })

    # regrouper par niveau horizontal Y
    words = sorted(words, key=lambda a: a["y"])

    groups = []
    for w in words:
        if not groups or abs(w["y"] - groups[-1]["y"]) > 18:
            groups.append({
                "y": w["y"],
                "items": [w]
            })
        else:
            groups[-1]["items"].append(w)
            groups[-1]["y"] = (groups[-1]["y"] + w["y"]) / 2

    rows = []

    for g in groups:
        items = sorted(g["items"], key=lambda a: a["x"])
        vals = [i["value"] for i in items]

        # chercher combinaison Pf, Pl, Em
        for i in range(len(vals) - 2):
            pf = vals[i]
            pl = vals[i + 1]
            em = vals[i + 2]

            if 0.1 <= pf <= 20 and 0.1 <= pl <= 30 and 10 <= em <= 20000:
                depth_raw = (g["y"] / h) * 15
                depth = snap_depth(depth_raw)

                if 0 < depth <= 15:
                    rows.append({
                        "depth": depth,
                        "pl": pl,
                        "em": em
                    })
                break

    # supprimer doublons mais garder la meilleure ligne
    final = {}

    for r in rows:
        d = r["depth"]
        final[d] = r

    result = [final[d] for d in sorted(final.keys())]

    return result, ocr_text(zone)


def make_excel(df):
    out = BytesIO()

    export = df[
        ["Nom du sondages", "x", "y", "Profondeur (m)", "Lithologie", "Pl* (MPa)", "Em (MPa)"]
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

        widths = {"A": 18, "B": 15, "C": 15, "D": 15, "E": 35, "F": 14, "G": 14}
        for c, w in widths.items():
            ws.column_dimensions[c].width = w

    out.seek(0)
    return out


if uploaded:
    if st.button("Extraire Excel"):
        doc = fitz.open(stream=uploaded.read(), filetype="pdf")
        all_rows = []

        progress = st.progress(0)

        for page_index, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))
            img = Image.open(BytesIO(pix.tobytes("png")))

            sondage, x, y = extract_header(img)
            data_rows, litho_text = extract_rows(img)
            litho1, litho2 = clean_litho(litho_text)

            for i, r in enumerate(data_rows):
                depth = r["depth"]
                litho = litho1 if depth <= 2.5 else litho2

                all_rows.append({
                    "Nom du sondages": sondage,
                    "x": x if i == 0 else "",
                    "y": y if i == 0 else "",
                    "Profondeur (m)": fr(depth),
                    "Lithologie": litho,
                    "Pl* (MPa)": fr(round(r["pl"], 3)),
                    "Em (MPa)": fr(round(r["em"], 1))
                })

            progress.progress((page_index + 1) / len(doc))

        df = pd.DataFrame(all_rows)
        st.dataframe(df, use_container_width=True)

        excel = make_excel(df)

        st.download_button(
            "Télécharger Excel",
            excel,
            "extraction_pressiometrique.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
