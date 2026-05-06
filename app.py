import re
from io import BytesIO
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image

st.set_page_config(page_title="PDF -> Tableau lithologie", layout="wide")


def clean_text(s: str) -> str:
    rep = {
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "à": "a", "â": "a",
        "î": "i", "ï": "i",
        "ô": "o",
        "ù": "u", "û": "u",
        "ç": "c",
    }
    s = s.lower()
    for a, b in rep.items():
        s = s.replace(a, b)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_label(label: str) -> str:
    t = clean_text(label)

    if "terre" in t and "veget" in t:
        return "Terre végétale"
    if "tuf" in t and "calc" in t:
        return "Tuf calcaire"
    if "roche" in t and "schist" in t:
        return "Roche schisteuse dure"
    if "roche" in t and "granit" in t:
        return "Roche granitique grise"
    if "roche" in t and "conglom" in t:
        return "Roche conglomératique"
    if "calcaire" in t and "mass" in t:
        return "Calcaire massif"
    if "calcaire" in t and "dur" in t:
        return "Calcaire dure"
    if "remblai" in t:
        return "Remblai"
    if "alluv" in t:
        return "Alluvions"
    if "argile" in t and "matrice" in t:
        return "Argile à matrice rocheuse"
    if "sable" in t and "matrice" in t:
        return "Sable à matrice rocheuse"
    if "argile" in t and "sable" in t:
        return "Sable argileux" if t.startswith("sable") else "Argile sableuse"
    if "argile" in t and "grave" in t:
        return "Argile graveleux"
    if "sable" in t and "grave" in t:
        return "Sable graveleux"
    if t == "argile":
        return "Argile"
    if t == "sable":
        return "Sable"

    return label.strip()


def parse_labels(lines):
    out = []
    i = 0

    while i < len(lines):
        cur = lines[i].strip()
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""

        cur_n = clean_text(cur)
        nxt_n = clean_text(nxt)
        combo = cur

        if cur_n == "roche" and any(k in nxt_n for k in ["schisteuse", "granitique", "conglomeratique", "conglomeratique"]):
            combo = cur + " " + nxt
            i += 1
        elif cur_n in ["argile a", "argile a matrice", "argile à", "argile à matrice"] and "matrice rocheuse" in nxt_n:
            combo = cur + " " + nxt
            i += 1
        elif cur_n in ["sable a", "sable a matrice", "sable à", "sable à matrice"] and "matrice rocheuse" in nxt_n:
            combo = cur + " " + nxt
            i += 1
        elif cur_n in ["argile", "sable"] and nxt_n in [
            "graveleux", "graveleuse", "granitique grise",
            "schisteuse dure", "conglomeratique", "dure"
        ]:
            combo = cur + " " + nxt
            i += 1

        lab = normalize_label(combo)
        if lab and (not out or out[-1] != lab):
            out.append(lab)

        i += 1

    return out


def extract_sondage_name(page_text: str, page_number: int) -> str:
    text = page_text.replace("\n", " ")

    patterns = [
        r"Sondage\s*[:\-]?\s*(SP[_\-\s]?(?:Rem|Reta)\d+)",
        r"(SP[_\-\s]?(?:Rem|Reta)\d+)",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            val = m.group(1)
            val = re.sub(r"[\s\-]+", "_", val)
            val = val.replace("SP__", "SP_")
            val = val.replace("rem", "Rem").replace("reta", "Reta")
            return val

    return f"Page_{page_number:03d}"


def render_page_to_array(doc, page_index: int, zoom: float = 2.0):
    page = doc[page_index]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return arr


def find_lithology_zone(arr: np.ndarray):
    # Zone approximative valable pour ce type de PDF Labotest
    return arr[430:1450, 240:470]


def detect_labels(arr: np.ndarray):
    zone = find_lithology_zone(arr)

    # OCR sur la sous-zone texte lithologie
    text_crop = Image.fromarray(zone[:, 35:180])
    txt = pytesseract.image_to_string(text_crop, config="--psm 6")
    raw_lines = [ln.strip() for ln in txt.splitlines() if re.search(r"[A-Za-zÀ-ÿ]", ln)]
    labels = parse_labels(raw_lines)

    return labels, txt


def detect_boundaries(arr: np.ndarray, labels: list[str]):
    zone = find_lithology_zone(arr)

    r = zone[:, :, 0].astype(int)
    g = zone[:, :, 1].astype(int)
    b = zone[:, :, 2].astype(int)

    # Les traits de séparation sont souvent rouges/roses
    metric = ((r - g) + (r - b)) / 2
    row_signal = metric.mean(axis=1)

    idx = [i for i, v in enumerate(row_signal) if v > 3.2]
    groups = []
    for i in idx:
        if not groups or i - groups[-1][-1] > 3:
            groups.append([i])
        else:
            groups[-1].append(i)

    centers = [round(sum(gp) / len(gp)) for gp in groups]

    bounds = []
    for c in centers:
        if c > 900:
            continue
        # calibration empirique pour ce type de page
        d = round(((c - 28.5) / 61) * 2) / 2
        if 0.2 <= d <= 14.8:
            if not bounds or abs(d - bounds[-1]) > 0.3:
                bounds.append(d)

    if labels and labels[0] == "Terre végétale":
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

    bounds = sorted(set(round(float(b) * 2) / 2 for b in bounds if 0 < b < 15))
    bounds = bounds[: max(len(labels) - 1, 0)]

    starts = [0] + bounds
    ends = bounds + [15]

    if len(starts) > len(labels):
        starts = starts[:len(labels)]
        ends = ends[:len(labels)]

    while len(starts) < len(labels):
        starts.append("")
        ends.append("")

    return starts, ends


def fix_split_labels(df: pd.DataFrame) -> pd.DataFrame:
    # Corrige les cas du genre:
    # "Argile" puis "graveleux"
    # "Sable" puis "graveleux"
    # ligne vide profondeur + "Roche ..."
    rows = []
    i = 0
    data = df.to_dict("records")

    while i < len(data):
        row = data[i]
        lith = str(row["Lithologie"]).strip()

        if i + 1 < len(data):
            nxt = data[i + 1]
            nxt_lith = str(nxt["Lithologie"]).strip()

            same_sondage = row["Sondage"] == nxt["Sondage"]

            if same_sondage and lith == "Argile" and nxt_lith == "graveleux":
                row["Lithologie"] = "Argile graveleux"
                row["Profondeur_fin (m)"] = nxt["Profondeur_fin (m)"]
                rows.append(row)
                i += 2
                continue

            if same_sondage and lith == "Sable" and nxt_lith == "graveleux":
                row["Lithologie"] = "Sable graveleux"
                row["Profondeur_fin (m)"] = nxt["Profondeur_fin (m)"]
                rows.append(row)
                i += 2
                continue

            if same_sondage and lith == "Sable" and nxt_lith == "argileux":
                row["Lithologie"] = "Sable argileux"
                row["Profondeur_fin (m)"] = nxt["Profondeur_fin (m)"]
                rows.append(row)
                i += 2
                continue

        if lith == "graveleux":
            i += 1
            continue

        rows.append(row)
        i += 1

    return pd.DataFrame(rows)


def extract_dataframe(pdf_bytes: bytes) -> pd.DataFrame:
    tmp_pdf = Path("tmp_upload.pdf")
    tmp_pdf.write_bytes(pdf_bytes)

    doc = fitz.open(str(tmp_pdf))
    rows = []

    for i in range(len(doc)):
        page = doc[i]
        page_text = page.get_text("text")
        sondage = extract_sondage_name(page_text, i + 1)

        arr = render_page_to_array(doc, i, zoom=2)
        labels, raw_ocr = detect_labels(arr)

        if not labels:
            continue

        starts, ends = detect_boundaries(arr, labels)

        for lith, z1, z2 in zip(labels, starts, ends):
            rows.append({
                "Sondage": sondage,
                "Profondeur_debut (m)": z1,
                "Profondeur_fin (m)": z2,
                "Lithologie": lith,
                "Page_source": i + 1,
                "OCR_lithologie": raw_ocr.replace("\n", " | "),
            })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = fix_split_labels(df)

    # Nettoyage final
    df["Profondeur_debut (m)"] = df["Profondeur_debut (m)"].astype(str)
    df["Profondeur_fin (m)"] = df["Profondeur_fin (m)"].astype(str)

    df = df[df["Lithologie"].astype(str).str.strip() != ""]
    df = df[~df["Lithologie"].astype(str).str.strip().eq("graveleux")]

    return df


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Lithologie")
        resume = df.groupby("Sondage", as_index=False).agg(
            Nb_couches=("Lithologie", "size")
        )
        resume.to_excel(writer, index=False, sheet_name="Resume")
    output.seek(0)
    return output.getvalue()


st.title("Extraction lithologie depuis PDF")
st.write("Importe uniquement le PDF des coupes. L'application génère : Sondage, Profondeur_debut (m), Profondeur_fin (m), Lithologie.")

pdf_file = st.file_uploader("PDF des coupes pressiométriques", type=["pdf"])

if st.button("Lancer l'extraction", type="primary"):
    if pdf_file is None:
        st.error("Charge d'abord le PDF.")
    else:
        try:
            with st.spinner("Extraction en cours..."):
                df = extract_dataframe(pdf_file.read())

            if df.empty:
                st.warning("Aucune donnée détectée.")
            else:
                st.success(f"Extraction terminée : {len(df)} lignes")
                st.dataframe(df[["Sondage", "Profondeur_debut (m)", "Profondeur_fin (m)", "Lithologie"]], use_container_width=True)

                excel_bytes = to_excel_bytes(df[["Sondage", "Profondeur_debut (m)", "Profondeur_fin (m)", "Lithologie"]])

                st.download_button(
                    "Télécharger l'Excel",
                    data=excel_bytes,
                    file_name="lithologie_sondages_depuis_pdf.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        except Exception as e:
            st.exception(e)
