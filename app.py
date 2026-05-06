import re
from io import BytesIO
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image, ImageFilter

st.set_page_config(page_title="Extraction lithologie", layout="wide")


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
    s = re.sub(r"[^a-z0-9_\-\s/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


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

    return normalize_spaces(label)


def merge_split_lines(lines):
    merged = []
    i = 0

    while i < len(lines):
        cur = normalize_spaces(lines[i])
        nxt = normalize_spaces(lines[i + 1]) if i + 1 < len(lines) else ""

        cur_n = clean_text(cur)
        nxt_n = clean_text(nxt)
        combo = cur

        if cur_n == "roche" and any(k in nxt_n for k in ["conglom", "schist", "granit"]):
            combo = f"{cur} {nxt}"
            i += 1
        elif cur_n in ["argile", "argile a", "argile a matrice"] and ("matrice" in nxt_n or "roche" in nxt_n):
            combo = f"{cur} {nxt}"
            i += 1
        elif cur_n in ["sable", "sable a", "sable a matrice"] and ("matrice" in nxt_n or "roche" in nxt_n):
            combo = f"{cur} {nxt}"
            i += 1
        elif cur_n == "argile" and "grave" in nxt_n:
            combo = f"{cur} {nxt}"
            i += 1
        elif cur_n == "sable" and ("grave" in nxt_n or "argileux" in nxt_n):
            combo = f"{cur} {nxt}"
            i += 1

        merged.append(combo)
        i += 1

    return merged


def parse_labels(lines):
    lines = merge_split_lines(lines)
    out = []
    for line in lines:
        lab = normalize_label(line)
        if lab and (not out or out[-1] != lab):
            out.append(lab)
    return out


def render_page_to_array(doc, page_index: int, zoom: float = 2.0):
    page = doc[page_index]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return arr


def detect_sondage_name_labotest(arr: np.ndarray) -> str:
    h, w = arr.shape[:2]

    # Ajuste ici si besoin
    x1 = int(w * 0.31)
    x2 = int(w * 0.53)
    y1 = int(h * 0.09)
    y2 = int(h * 0.17)

    crop = arr[y1:y2, x1:x2]
    base_img = Image.fromarray(crop).convert("L")

    variants = [
        base_img,
        base_img.resize((base_img.width * 4, base_img.height * 4)),
        base_img.point(lambda p: 255 if p > 170 else 0).resize((base_img.width * 4, base_img.height * 4)),
        base_img.point(lambda p: 255 if p > 145 else 0).resize((base_img.width * 4, base_img.height * 4)),
        base_img.filter(ImageFilter.SHARPEN).resize((base_img.width * 4, base_img.height * 4)),
    ]

    psm_modes = [6, 7, 11]

    for img in variants:
        for psm in psm_modes:
            txt = pytesseract.image_to_string(img, config=f"--psm {psm}")
            txt = txt.replace("\n", " ").strip()

            m = re.search(r"Sondage\s*:\s*([A-Za-z0-9_\-/]+)", txt, flags=re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if val and clean_text(val) != "objet":
                    return val

            m = re.search(r"(SP[_\-]?(?:Rem|Reta)[_\-]?\d+)", txt, flags=re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if val:
                    return val

    return ""


def labotest_lithology_zone(arr: np.ndarray):
    return arr[430:1450, 240:470]


def extract_labels_labotest(arr: np.ndarray):
    zone = labotest_lithology_zone(arr)
    text_crop = Image.fromarray(zone[:, 35:190])
    txt = pytesseract.image_to_string(text_crop, config="--psm 6")
    raw_lines = [ln.strip() for ln in txt.splitlines() if re.search(r"[A-Za-zÀ-ÿ]", ln)]
    return parse_labels(raw_lines)


def extract_depths_labotest(arr: np.ndarray, labels: list[str]):
    zone = labotest_lithology_zone(arr)

    r = zone[:, :, 0].astype(int)
    g = zone[:, :, 1].astype(int)
    b = zone[:, :, 2].astype(int)

    metric = ((r - g) + (r - b)) / 2
    row_signal = metric.mean(axis=1)

    idx = [i for i, v in enumerate(row_signal) if v > 3.2]
    groups = []
    for j in idx:
        if not groups or j - groups[-1][-1] > 3:
            groups.append([j])
        else:
            groups[-1].append(j)

    centers = [round(sum(gp) / len(gp)) for gp in groups]

    bounds = []
    for c in centers:
        if c > 900:
            continue
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
        elif bounds and bounds[0] < 1.0:
            bounds[0] = 0.5

    bounds = sorted(set(round(float(b) * 2) / 2 for b in bounds if 0 < b < 15))
    bounds = bounds[:max(len(labels) - 1, 0)]

    starts = [0] + bounds
    ends = bounds + [15]

    if len(starts) > len(labels):
        starts = starts[:len(labels)]
        ends = ends[:len(labels)]

    while len(starts) < len(labels):
        starts.append("")
        ends.append("")

    return starts, ends


def fix_final_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    data = df.to_dict("records")
    i = 0

    while i < len(data):
        row = data[i]
        lith = str(row["Lithologie"]).strip()
        lith_n = clean_text(lith)

        if i + 1 < len(data):
            nxt = data[i + 1]
            same_sondage = row["Sondage"] == nxt["Sondage"]
            nxt_lith = str(nxt["Lithologie"]).strip()
            nxt_lith_n = clean_text(nxt_lith)

            if same_sondage:
                if lith_n == "roche" and "conglom" in nxt_lith_n:
                    row["Lithologie"] = "Roche conglomératique"
                    row["Profondeur_fin (m)"] = nxt["Profondeur_fin (m)"]
                    rows.append(row)
                    i += 2
                    continue

                if lith_n == "roche" and "schist" in nxt_lith_n:
                    row["Lithologie"] = "Roche schisteuse dure"
                    row["Profondeur_fin (m)"] = nxt["Profondeur_fin (m)"]
                    rows.append(row)
                    i += 2
                    continue

                if lith_n == "roche" and "granit" in nxt_lith_n:
                    row["Lithologie"] = "Roche granitique grise"
                    row["Profondeur_fin (m)"] = nxt["Profondeur_fin (m)"]
                    rows.append(row)
                    i += 2
                    continue

                if "argile" in lith_n and ("matrice" in nxt_lith_n or "roche" in nxt_lith_n):
                    row["Lithologie"] = "Argile à matrice rocheuse"
                    row["Profondeur_fin (m)"] = nxt["Profondeur_fin (m)"]
                    rows.append(row)
                    i += 2
                    continue

                if "sable" in lith_n and ("matrice" in nxt_lith_n or "roche" in nxt_lith_n):
                    row["Lithologie"] = "Sable à matrice rocheuse"
                    row["Profondeur_fin (m)"] = nxt["Profondeur_fin (m)"]
                    rows.append(row)
                    i += 2
                    continue

                if lith_n == "argile" and "grave" in nxt_lith_n:
                    row["Lithologie"] = "Argile graveleux"
                    row["Profondeur_fin (m)"] = nxt["Profondeur_fin (m)"]
                    rows.append(row)
                    i += 2
                    continue

                if lith_n == "sable" and "grave" in nxt_lith_n:
                    row["Lithologie"] = "Sable graveleux"
                    row["Profondeur_fin (m)"] = nxt["Profondeur_fin (m)"]
                    rows.append(row)
                    i += 2
                    continue

        if lith_n in ["graveleux", "graveleuse", "matrice roche", "matrice rocheuse", "conglomeratic", "conglomeratique"]:
            i += 1
            continue

        rows.append(row)
        i += 1

    out = pd.DataFrame(rows)
    out = out[out["Lithologie"].astype(str).str.strip() != ""]
    return out.reset_index(drop=True)


def normalize_depth_value(v):
    if v == "" or v is None:
        return ""
    try:
        x = float(v)
        if abs(x - round(x)) < 1e-9:
            return int(round(x))
        return round(x, 2)
    except Exception:
        return v


def extract_dataframe(pdf_bytes: bytes):
    tmp_pdf = Path("tmp_upload.pdf")
    tmp_pdf.write_bytes(pdf_bytes)

    doc = fitz.open(str(tmp_pdf))
    rows = []
    undetected_pages = []

    for i in range(len(doc)):
        arr = render_page_to_array(doc, i, zoom=2)

        sondage = detect_sondage_name_labotest(arr)
        if not sondage:
            undetected_pages.append(i + 1)
            sondage = f"PAGE_{i+1:03d}"

        labels = extract_labels_labotest(arr)
        if not labels:
            continue

        starts, ends = extract_depths_labotest(arr, labels)

        for lith, z1, z2 in zip(labels, starts, ends):
            rows.append({
                "Sondage": sondage,
                "Lithologie": lith,
                "Profondeur_debut (m)": normalize_depth_value(z1),
                "Profondeur_fin (m)": normalize_depth_value(z2),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = fix_final_dataframe(df)

    return df, undetected_pages


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Lithologie")
    output.seek(0)
    return output.getvalue()


st.title("Extraction lithologie")

pdf_file = st.file_uploader("", type=["pdf"])

if st.button("Lancer l'extraction", type="primary"):
    if pdf_file is None:
        st.error("Charge d'abord le PDF.")
    else:
        try:
            with st.spinner("Extraction en cours..."):
                df, undetected_pages = extract_dataframe(pdf_file.read())

            if undetected_pages:
                st.warning(f"Nom du sondage non détecté sur les pages : {undetected_pages}")

            if df.empty:
                st.warning("Aucune donnée détectée.")
            else:
                cols = ["Sondage", "Lithologie", "Profondeur_debut (m)", "Profondeur_fin (m)"]
                st.success(f"Extraction terminée : {len(df)} lignes")
                st.dataframe(df[cols], use_container_width=True)

                st.download_button(
                    "Télécharger l'Excel",
                    data=to_excel_bytes(df[cols]),
                    file_name="lithologie_labotest.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        except Exception as e:
            st.exception(e)
