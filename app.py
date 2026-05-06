import re
from io import BytesIO
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Extraction lithologie sondages", layout="wide")

def normalize_label(label: str) -> str:
    t = label.lower()
    rep = {'é':'e','è':'e','ê':'e','à':'a','û':'u','î':'i','ï':'i','ô':'o','ç':'c'}
    for a, b in rep.items():
        t = t.replace(a, b)
    t = re.sub(r'[^a-z\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()

    if 'terre' in t and 'veget' in t:
        return 'Terre végétale'
    if 'tuf' in t and 'calc' in t:
        return 'Tuf calcaire'
    if 'roche' in t and 'schist' in t:
        return 'Roche schisteuse dure'
    if 'roche' in t and 'granit' in t:
        return 'Roche granitique grise'
    if 'roche' in t and 'conglom' in t:
        return 'Roche conglomératique'
    if 'calcaire' in t and 'mass' in t:
        return 'Calcaire massif'
    if 'calcaire' in t and 'dur' in t:
        return 'Calcaire dure'
    if 'remblai' in t:
        return 'Remblai'
    if 'alluv' in t:
        return 'Alluvions'
    if t == 'argile' or ('argile' in t and 'grave' not in t and 'matrice' not in t and 'sable' not in t):
        return 'Argile'
    if 'argile' in t and 'sable' in t:
        return 'Sable argileux' if t.startswith('sable') else 'Argile sableuse'
    if 'argile' in t and 'grave' in t:
        return 'Argile graveleux'
    if 'sable' in t and 'grave' in t:
        return 'Sable graveleux'
    if 'argile' in t and 'matrice' in t:
        return 'Argile à matrice rocheuse'
    if 'sable' in t and 'matrice' in t:
        return 'Sable à matrice rocheuse'
    if t == 'sable':
        return 'Sable'
    return label.strip()

def parse_labels(lines):
    out = []
    i = 0
    while i < len(lines):
        cur = lines[i].strip()
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ''
        cur_n = cur.lower()
        nxt_n = nxt.lower()
        combo = cur

        if cur_n == 'roche' and any(k in nxt_n for k in ['schisteuse', 'granitique', 'conglomer', 'conglom']):
            combo = cur + ' ' + nxt
            i += 1
        elif cur_n in ['argile a', 'argile à', 'sable a', 'sable à'] and 'matrice rocheuse' in nxt_n:
            combo = cur + ' ' + nxt
            i += 1
        elif cur_n in ['argile', 'sable'] and nxt_n in [
            'a matrice rocheuse', 'à matrice rocheuse',
            'granitique grise', 'schisteuse dure',
            'conglomeratique', 'conglomératique', 'dure'
        ]:
            combo = cur + ' ' + nxt
            i += 1

        lab = normalize_label(combo)
        if lab and (not out or out[-1] != lab):
            out.append(lab)
        i += 1
    return out

def render_page_to_array(doc, page_index, zoom=2):
    page = doc[page_index]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return arr

def get_page_data(arr):
    lith_img = Image.fromarray(arr[430:1450, 280:460])
    txt = pytesseract.image_to_string(lith_img, config='--psm 6')
    lines = [ln.strip() for ln in txt.splitlines() if re.search(r'[A-Za-zÀ-ÿ]', ln)]
    labels = parse_labels(lines)

    crop = arr[430:1450, 240:470]
    r, g, b = [crop[:, :, i].astype(int) for i in range(3)]
    metric = ((r - g) + (r - b)) / 2
    rows = metric.mean(axis=1)

    idx = [i for i, v in enumerate(rows) if v > 3.2]
    groups = []
    for i in idx:
        if not groups or i - groups[-1][-1] > 3:
            groups.append([i])
        else:
            groups[-1].append(i)
    centers = [round(sum(g) / len(g)) for g in groups]

    bounds = []
    for c in centers:
        if c > 900:
            continue
        d = round(((c - 28.5) / 61) * 2) / 2
        if 0.2 <= d <= 14.8:
            if not bounds or abs(d - bounds[-1]) > 0.3:
                bounds.append(d)

    if labels and labels[0] == 'Terre végétale':
        if not bounds or bounds[0] > 1.0:
            small = [c for c in centers if c < 120]
            if small:
                c = min(small)
                d = 0.5 if c < 45 else round(((c - 28.5) / 61) * 2) / 2
                if d < 1.0:
                    bounds = [d] + bounds
        else:
            if bounds[0] < 1.0:
                bounds[0] = 0.5

    bounds = sorted(set([round(float(b) * 2) / 2 for b in bounds if 0 < b < 15]))
    bounds = bounds[:max(len(labels) - 1, 0)]

    starts = [0] + bounds
    ends = bounds + [15]

    if len(starts) > len(labels):
        starts = starts[:len(labels)]
        ends = ends[:len(labels)]
    while len(starts) < len(labels):
        starts.append("")
        ends.append("")

    return labels, starts, ends, txt

def extract_dataframe(pdf_bytes, coords_df):
    tmp_pdf = Path("tmp_upload.pdf")
    tmp_pdf.write_bytes(pdf_bytes)
    doc = fitz.open(str(tmp_pdf))

    rows = []
    n = min(len(doc), len(coords_df))

    for i in range(n):
        arr = render_page_to_array(doc, i, zoom=2)
        labels, starts, ends, raw_ocr = get_page_data(arr)

        sondage = coords_df.iloc[i]["Sondage"]
        x = coords_df.iloc[i]["X"]
        y = coords_df.iloc[i]["Y"]

        for lab, z1, z2 in zip(labels, starts, ends):
            rows.append({
                "Sondage": sondage,
                "X": x,
                "Y": y,
                "Profondeur_debut (m)": z1,
                "Profondeur_fin (m)": z2,
                "Lithologie": lab,
                "Page_source": i + 1,
                "OCR_lithologie": raw_ocr.replace("\n", " | "),
            })

    return pd.DataFrame(rows)

def to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Lithologie")
        resume = df.groupby("Sondage", as_index=False).agg(
            X=("X", "first"),
            Y=("Y", "first"),
            Nb_couches=("Lithologie", "size"),
        )
        resume.to_excel(writer, index=False, sheet_name="Resume")
    output.seek(0)
    return output.getvalue()

st.title("Extraction lithologie des sondages")
st.write("Importe le PDF des coupes + un Excel des coordonnées (colonnes : Sondage, X, Y).")

col1, col2 = st.columns(2)
with col1:
    pdf_file = st.file_uploader("PDF des coupes", type=["pdf"])
with col2:
    coords_file = st.file_uploader("Excel coordonnées", type=["xlsx", "xls"])

st.caption("Cette version suit la logique utilisée sur ton PDF : OCR de la colonne lithologie + détection des traits horizontaux sur l’axe vertical.")

if st.button("Lancer l'extraction", type="primary"):
    if pdf_file is None or coords_file is None:
        st.error("Charge d'abord le PDF et l'Excel des coordonnées.")
    else:
        try:
            coords_df = pd.read_excel(coords_file)
            expected = {"Sondage", "X", "Y"}
            if not expected.issubset(set(coords_df.columns)):
                st.error("Le fichier coordonnées doit contenir : Sondage, X, Y")
            else:
                with st.spinner("Extraction en cours..."):
                    df = extract_dataframe(pdf_file.read(), coords_df)

                st.success(f"Extraction terminée : {len(df)} lignes")
                st.dataframe(df, use_container_width=True)

                excel_bytes = to_excel_bytes(df)
                st.download_button(
                    "Télécharger l'Excel",
                    data=excel_bytes,
                    file_name="lithologie_sondages_depuis_pdf.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        except Exception as e:
            st.exception(e)
