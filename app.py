
import re
from io import BytesIO
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Extraction lithologie PDF", layout="wide")


# =========================================================
# TEXTE
# =========================================================
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
    s = re.sub(r"[^a-z0-9\-\s_/]", " ", s)
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
    if t == "tirs":
        return "Tirs"
    if "gres" in t and "lumachell" in t:
        return "Grès lumachellique"
    if "marne" in t and "verd" in t:
        return "Marne verdâtre avec traces bariolée d'oxydations"
    if "schiste" in t and "fractur" in t:
        return "Schiste dur fracturé à fracturations sub-horizontales"

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

        elif "marne" in cur_n and i + 1 < len(lines):
            if any(x in nxt_n for x in ["bariolee", "oxydation", "traces"]):
                combo = f"{cur} {nxt}"
                i += 1

        elif "schiste" in cur_n and i + 1 < len(lines):
            if "fractur" in nxt_n:
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


# =========================================================
# NOM EXACT DU SONDAGE
# =========================================================
def extract_exact_sondage_name(page_text: str, page_number: int) -> str:
    txt = page_text.replace("\n", " ")

    patterns = [
        r"Sondage\s+pressiometrique\s+Menard\s*[:\-]?\s*([A-Za-z0-9_\-\/]+)",
        r"Sondage\s+pressiom[eé]trique\s+M[eé]nard\s*[:\-]?\s*([A-Za-z0-9_\-\/]+)",
        r"Sondage\s*[:\-]?\s*(SP[_\-\s]?(?:Rem|Reta)\d+)",
        r"(SP[_\-\s]?(?:Rem|Reta)\d+)",
        r"(T\d+\-SCP\-EXE\-\d+)",
        r"(T\d+\-[A-Za-z0-9\-]+)",
    ]

    for pat in patterns:
        m = re.search(pat, txt, flags=re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            val = val.replace(" ", "_")
            val = re.sub(r"_+", "_", val)

            # Normalisation SP_Rem / SP_Reta
            val = re.sub(r"(?i)^sp[_\- ]?rem[_\- ]?(\d+)$", lambda x: f"SP_Rem{int(x.group(1)):03d}", val)
            val = re.sub(r"(?i)^sp[_\- ]?reta[_\- ]?(\d+)$", lambda x: f"SP_Reta{int(x.group(1)):03d}", val)

            # Garde LPEE tel quel
            return val

    return f"Sondage_{page_number:03d}"


# =========================================================
# IMAGE PAGE
# =========================================================
def render_page_to_array(doc, page_index: int, zoom: float = 2.0):
    page = doc[page_index]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return arr


# =========================================================
# DETECTION MODE PDF
# =========================================================
def detect_pdf_type(page_text: str) -> str:
    t = clean_text(page_text)

    if "labotest" in t:
        return "LABOTEST"
    if "laboratoire public d essais" in t or "centre experimental des sols" in t or "jean lutz" in t:
        return "LPEE"
    return "GENERIC"


# =========================================================
# LABOTEST
# =========================================================
def labotest_lithology_zone(arr: np.ndarray):
    return arr[430:1450, 240:470]


def extract_labels_labotest(arr: np.ndarray):
    zone = labotest_lithology_zone(arr)
    text_crop = Image.fromarray(zone[:, 35:190])
    txt = pytesseract.image_to_string(text_crop, config="--psm 6")
    raw_lines = [ln.strip() for ln in txt.splitlines() if re.search(r"[A-Za-zÀ-ÿ]", ln)]
    labels = parse_labels(raw_lines)
    return labels, txt


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
        elif bounds[0] < 1.0:
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


# =========================================================
# LPEE / CES
# =========================================================
def extract_lpee_breaks_from_text(page_text: str):
    """
    Pour le type LPEE, le texte parsé contient souvent:
    1.50
    12.50
    15.00
    22.00
    Lithologie
    Tirs
    Grès lumachellique
    ...
    """
    lines = [normalize_spaces(x) for x in page_text.splitlines() if normalize_spaces(x)]

    # profondeurs
    depth_candidates = []
    after_prof = False
    for ln in lines:
        ln_n = clean_text(ln)

        if "prof sous" in ln_n or "tn" in ln_n:
            after_prof = True
            continue

        if after_prof:
            m = re.fullmatch(r"(\d+(?:\.\d+)?)", ln.strip())
            if m:
                val = float(m.group(1))
                if 0 < val <= 100:
                    depth_candidates.append(val)

        if "lithologie" in ln_n:
            break

    # garde des profondeurs strictement croissantes
    depths = []
    for d in depth_candidates:
        if not depths or d > depths[-1]:
            depths.append(d)

    # lithologies
    labels = []
    start_labels = False
    stop_words = [
        "tubage", "couronne", "pression de fluage",
        "pression limite", "module pressiometrique",
        "e/pl", "pf", "pl", "em"
    ]

    for ln in lines:
        ln_n = clean_text(ln)

        if ln_n == "lithologie":
            start_labels = True
            continue

        if start_labels:
            if any(w in ln_n for w in stop_words):
                break

            if re.search(r"[A-Za-zÀ-ÿ]", ln):
                labels.append(ln)

    labels = parse_labels(labels)

    return depths, labels


def extract_lpee_data(page_text: str):
    total_depth = None

    m = re.search(r"Profondeur\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*m", page_text, flags=re.IGNORECASE)
    if m:
        total_depth = float(m.group(1))

    breaks, labels = extract_lpee_breaks_from_text(page_text)

    if not labels:
        return [], [], [], ""

    if total_depth is None:
        total_depth = breaks[-1] if breaks else 15.0

    # Ex: labels = 4 couches, breaks = [1.5, 12.5, 15, 22]
    # => starts [0,1.5,12.5,15], ends [1.5,12.5,15,22]
    if len(breaks) >= len(labels):
        ends = breaks[:len(labels)]
        starts = [0.0] + ends[:-1]
    else:
        # fallback
        starts = [0.0]
        for _ in range(len(labels) - 1):
            starts.append("")
        ends = ["" for _ in labels]
        if ends:
            ends[-1] = total_depth

    # complète la dernière fin si besoin
    if len(ends) == len(labels) and ends[-1] == "":
        ends[-1] = total_depth

    return labels, starts, ends, "LPEE_TEXT"


# =========================================================
# GENERIQUE
# =========================================================
def generic_lithology_zone(arr: np.ndarray):
    h, w = arr.shape[:2]
    x1 = int(w * 0.07)
    x2 = int(w * 0.38)
    y1 = int(h * 0.18)
    y2 = int(h * 0.92)
    return arr[y1:y2, x1:x2]


def extract_generic_data(arr: np.ndarray):
    zone = generic_lithology_zone(arr)

    txt = pytesseract.image_to_string(Image.fromarray(zone), config="--psm 6")
    raw_lines = [ln.strip() for ln in txt.splitlines() if re.search(r"[A-Za-zÀ-ÿ]", ln)]
    labels = parse_labels(raw_lines)

    gray = np.mean(zone[:, :, :3], axis=2)
    diff = np.abs(np.diff(gray, axis=0)).mean(axis=1)

    idx = [i for i, v in enumerate(diff) if v > np.percentile(diff, 96)]
    groups = []
    for j in idx:
        if not groups or j - groups[-1][-1] > 4:
            groups.append([j])
        else:
            groups[-1].append(j)

    centers = [round(sum(gp) / len(gp)) for gp in groups]

    bounds = []
    if len(zone) > 0:
        for c in centers:
            ratio = c / len(zone)
            d = round((ratio * 15) * 2) / 2
            if 0.2 <= d <= 14.8:
                if not bounds or abs(d - bounds[-1]) > 0.5:
                    bounds.append(d)

    bounds = bounds[: max(len(labels) - 1, 0)]

    starts = [0] + bounds
    ends = bounds + [15]

    if len(starts) > len(labels):
        starts = starts[:len(labels)]
        ends = ends[:len(labels)]

    while len(starts) < len(labels):
        starts.append("")
        ends.append("")

    return labels, starts, ends, txt


# =========================================================
# POST TRAITEMENT
# =========================================================
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

                if lith_n == "sable" and "argileux" in nxt_lith_n:
                    row["Lithologie"] = "Sable argileux"
                    row["Profondeur_fin (m)"] = nxt["Profondeur_fin (m)"]
                    rows.append(row)
                    i += 2
                    continue

        if lith_n in [
            "graveleux", "graveleuse", "matrice roche", "matrice rocheuse",
            "conglomeratic", "conglomeratique"
        ]:
            i += 1
            continue

        rows.append(row)
        i += 1

    out = pd.DataFrame(rows)
    out = out[out["Lithologie"].astype(str).str.strip() != ""]
    out = out.reset_index(drop=True)
    return out


def normalize_depth_value(v):
    if v == "" or v is None:
        return ""
    try:
        x = float(v)
        if abs(x - round(x)) < 1e-9:
            return int(round(x))
        return round(x, 2)
    except:
        return v


# =========================================================
# EXTRACTION GLOBALE
# =========================================================
def extract_dataframe(pdf_bytes: bytes) -> pd.DataFrame:
    tmp_pdf = Path("tmp_upload.pdf")
    tmp_pdf.write_bytes(pdf_bytes)

    doc = fitz.open(str(tmp_pdf))
    rows = []

    for i in range(len(doc)):
        page = doc[i]
        page_text = page.get_text("text")
        sondage = extract_exact_sondage_name(page_text, i + 1)
        pdf_type = detect_pdf_type(page_text)

        if pdf_type == "LPEE":
            labels, starts, ends, source = extract_lpee_data(page_text)

        else:
            arr = render_page_to_array(doc, i, zoom=2)

            if pdf_type == "LABOTEST":
                labels, raw_ocr = extract_labels_labotest(arr)
                if labels:
                    starts, ends = extract_depths_labotest(arr, labels)
                else:
                    starts, ends = [], []
                source = raw_ocr
            else:
                labels, starts, ends, source = extract_generic_data(arr)

        if not labels:
            continue

        for lith, z1, z2 in zip(labels, starts, ends):
            rows.append({
                "Sondage": sondage,
                "Lithologie": lith,
                "Profondeur_debut (m)": normalize_depth_value(z1),
                "Profondeur_fin (m)": normalize_depth_value(z2),
                "Type_PDF": pdf_type,
            })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = fix_final_dataframe(df)
    return df


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Lithologie")
        resume = df.groupby("Sondage", as_index=False).agg(
            Nb_couches=("Lithologie", "size"),
            Type_PDF=("Type_PDF", "first"),
        )
        resume.to_excel(writer, index=False, sheet_name="Resume")
    output.seek(0)
    return output.getvalue()


# =========================================================
# INTERFACE
# =========================================================
st.title("Extraction lithologie depuis PDF")
st.write("Importe un PDF Labotest ou LPEE/CES. Le nom du sondage sera gardé tel quel.")

pdf_file = st.file_uploader("PDF", type=["pdf"])

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
                show_cols = [
                    "Sondage",
                    "Lithologie",
                    "Profondeur_debut (m)",
                    "Profondeur_fin (m)",
                    "Type_PDF",
                ]
                st.success(f"Extraction terminée : {len(df)} lignes")
                st.dataframe(df[show_cols], use_container_width=True)

                excel_bytes = to_excel_bytes(df[show_cols])

                st.download_button(
                    "Télécharger l'Excel",
                    data=excel_bytes,
                    file_name="lithologie_sondages_mixte.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        except Exception as e:
            st.exception(e)
