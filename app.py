import streamlit as st
import pandas as pd
import fitz
import pytesseract
import re
from PIL import Image, ImageDraw
from io import BytesIO
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.set_page_config(page_title="Extraction coupes pressiométriques", layout="wide")
st.title("Extraction PDF pressiométrique avec calibration")

uploaded = st.file_uploader("Importer PDF", type=["pdf"])

def crop(img, box):
    w, h = img.size
    x1, y1, x2, y2 = box
    return img.crop((int(w*x1), int(h*y1), int(w*x2), int(h*y2)))

def draw_box(img, box, label):
    im = img.copy()
    d = ImageDraw.Draw(im)
    w, h = im.size
    x1, y1, x2, y2 = box
    pts = (int(w*x1), int(h*y1), int(w*x2), int(h*y2))
    d.rectangle(pts, outline="red", width=5)
    d.text((pts[0], pts[1]-25), label, fill="red")
    return im

def ocr_text(img, psm=6):
    return pytesseract.image_to_string(img, lang="fra+eng", config=f"--psm {psm}")

def ocr_data(img, psm=6):
    return pytesseract.image_to_data(
        img,
        lang="fra+eng",
        config=f"--psm {psm}",
        output_type=pytesseract.Output.DATAFRAME
    )

def to_float(x):
    try:
        return float(str(x).replace(",", "."))
    except:
        return None

def fr(x):
    if x is None or x == "":
        return ""
    return str(x).replace(".", ",")

def extract_header(img, header_box):
    z = crop(img, header_box)
    t = ocr_text(z, 6)

    sondage = ""
    x = ""
    y = ""

    m = re.search(r"Sondage\s*[:\-]?\s*([A-Z]{1,5}[_\-]?[A-Za-z0-9]+)", t, re.I)
    if m:
        sondage = m.group(1).replace("-", "_").strip()

    mx = re.search(r"X\s*[:\-]?\s*(\d+[.,]\d+)", t)
    my = re.search(r"Y\s*[:\-]?\s*(\d+[.,]\d+)", t)

    if mx:
        x = fr(mx.group(1))
    if my:
        y = fr(my.group(1))

    return sondage, x, y, t

def extract_numbers_with_depth(zone, depth_max, vmin, vmax):
    df = ocr_data(zone, 6)
    df = df.dropna(subset=["text"])
    h = zone.size[1]
    values = []

    for _, r in df.iterrows():
        txt = str(r["text"]).strip()
        txt = txt.replace("O", "0").replace("o", "0").replace("|", "1").replace("l", "1")
        m = re.search(r"\d+[.,]\d+", txt)
        if not m:
            continue

        val = to_float(m.group(0))
        if val is None:
            continue

        if vmin <= val <= vmax:
            cy = r["top"] + r["height"] / 2
            depth = round((cy / h) * depth_max, 2)
            values.append({"depth": depth, "value": val})

    values = sorted(values, key=lambda x: x["depth"])

    clean = []
    for v in values:
        if not clean or abs(v["depth"] - clean[-1]["depth"]) > 0.25:
            clean.append(v)

    return clean

def extract_lithology_points(zone, depth_max):
    df = ocr_data(zone, 6)
    df = df.dropna(subset=["text"])
    h = zone.size[1]

    words = []
    for _, r in df.iterrows():
        txt = str(r["text"]).strip()
        if not re.search(r"[A-Za-zéèêàâîïôùûç]", txt):
            continue
        if txt.lower() in ["lithologie", "profondeur", "labotest", "m"]:
            continue
        cy = r["top"] + r["height"] / 2
        depth = round((cy / h) * depth_max, 2)
        words.append((depth, txt))

    words = sorted(words, key=lambda x: x[0])

    groups = []
    for depth, word in words:
        if not groups or abs(depth - groups[-1]["depth"]) > 0.9:
            groups.append({"depth": depth, "words": [word]})
        else:
            groups[-1]["words"].append(word)
            groups[-1]["depth"] = round((groups[-1]["depth"] + depth) / 2, 2)

    return [{"depth": g["depth"], "lithologie": " ".join(g["words"])} for g in groups]

def lithology_for_depth(depth, litho_points):
    if not litho_points:
        return ""

    litho_points = sorted(litho_points, key=lambda x: x["depth"])
    nearest = min(litho_points, key=lambda x: abs(x["depth"] - depth))
    return nearest["lithologie"]

def merge_pl_em(pls, ems):
    rows = []
    used = set()

    for pl in pls:
        em_val = ""
        if ems:
            candidates = [(i, e) for i, e in enumerate(ems) if i not in used]
            if candidates:
                i_best, best = min(candidates, key=lambda x: abs(x[1]["depth"] - pl["depth"]))
                if abs(best["depth"] - pl["depth"]) <= 0.8:
                    em_val = best["value"]
                    used.add(i_best)

        rows.append({
            "depth": pl["depth"],
            "pl": pl["value"],
            "em": em_val
        })

    return rows

def make_excel(df):
    out = BytesIO()
    export = df[["Nom du sondages", "x", "y", "Profondeur (m)", "Lithologie", "Pl* (MPa)", "Em (MPa)"]]

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

        for col, width in {"A":18,"B":15,"C":15,"D":15,"E":35,"F":14,"G":14}.items():
            ws.column_dimensions[col].width = width

    out.seek(0)
    return out

if uploaded:
    doc = fitz.open(stream=uploaded.read(), filetype="pdf")

    page_preview = st.number_input("Page test", min_value=1, max_value=len(doc), value=1)
    depth_max = st.number_input("Profondeur max de la coupe", value=15.0, step=0.5)

    page = doc[page_preview - 1]
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
    img = Image.open(BytesIO(pix.tobytes("png")))

    st.subheader("Calibration des zones")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Zone entête")
        hx1 = st.slider("header x1", 0.0, 1.0, 0.30, 0.01)
        hy1 = st.slider("header y1", 0.0, 1.0, 0.12, 0.01)
        hx2 = st.slider("header x2", 0.0, 1.0, 0.98, 0.01)
        hy2 = st.slider("header y2", 0.0, 1.0, 0.25, 0.01)

        st.write("Zone lithologie")
        lx1 = st.slider("litho x1", 0.0, 1.0, 0.13, 0.01)
        ly1 = st.slider("litho y1", 0.0, 1.0, 0.28, 0.01)
        lx2 = st.slider("litho x2", 0.0, 1.0, 0.39, 0.01)
        ly2 = st.slider("litho y2", 0.0, 1.0, 0.90, 0.01)

    with col2:
        st.write("Zone Pl")
        px1 = st.slider("Pl x1", 0.0, 1.0, 0.63, 0.01)
        py1 = st.slider("Pl y1", 0.0, 1.0, 0.28, 0.01)
        px2 = st.slider("Pl x2", 0.0, 1.0, 0.78, 0.01)
        py2 = st.slider("Pl y2", 0.0, 1.0, 0.90, 0.01)

        st.write("Zone Em")
        ex1 = st.slider("Em x1", 0.0, 1.0, 0.78, 0.01)
        ey1 = st.slider("Em y1", 0.0, 1.0, 0.28, 0.01)
        ex2 = st.slider("Em x2", 0.0, 1.0, 0.97, 0.01)
        ey2 = st.slider("Em y2", 0.0, 1.0, 0.90, 0.01)

    header_box = (hx1, hy1, hx2, hy2)
    litho_box = (lx1, ly1, lx2, ly2)
    pl_box = (px1, py1, px2, py2)
    em_box = (ex1, ey1, ex2, ey2)

    preview = img.copy()
    for box, lab in [(header_box, "HEADER"), (litho_box, "LITHO"), (pl_box, "PL"), (em_box, "EM")]:
        preview = draw_box(preview, box, lab)

    st.image(preview, caption="Ajuste les zones avec les sliders", use_container_width=True)

    if st.button("Extraire tout le PDF"):
        all_rows = []
        progress = st.progress(0)

        for idx, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            img = Image.open(BytesIO(pix.tobytes("png")))

            sondage, x, y, header_text = extract_header(img, header_box)

            litho_zone = crop(img, litho_box)
            pl_zone = crop(img, pl_box)
            em_zone = crop(img, em_box)

            lithos = extract_lithology_points(litho_zone, depth_max)
            pls = extract_numbers_with_depth(pl_zone, depth_max, 0.1, 20)
            ems = extract_numbers_with_depth(em_zone, depth_max, 1, 10000)

            merged = merge_pl_em(pls, ems)

            for i, r in enumerate(merged):
                d = r["depth"]
                all_rows.append({
                    "Page PDF": idx + 1,
                    "Nom du sondages": sondage,
                    "x": x if i == 0 else "",
                    "y": y if i == 0 else "",
                    "Profondeur (m)": fr(round(d, 2)),
                    "Lithologie": lithology_for_depth(d, lithos),
                    "Pl* (MPa)": fr(round(r["pl"], 3)),
                    "Em (MPa)": fr(round(r["em"], 1)) if r["em"] != "" else ""
                })

            progress.progress((idx + 1) / len(doc))

        df = pd.DataFrame(all_rows)

        st.subheader("Résultat extrait — tu peux corriger avant Excel")
        edited = st.data_editor(df, use_container_width=True, num_rows="dynamic")

        excel = make_excel(edited)
        st.download_button(
            "Télécharger Excel",
            excel,
            "extraction_pressiometrique.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
