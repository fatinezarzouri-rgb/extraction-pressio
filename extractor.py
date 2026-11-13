
"""
Extraction générale des coupes d'essais pressiométriques (PDF -> tableau).
 
Principe (aucune coordonnée fixe, aucun nom de lithologie codé en dur) :
1. On lit les mots du PDF avec leurs positions (texte natif, sinon OCR).
2. On repère les en-têtes "Pf", "Pl", "Em" -> cela donne les 3 colonnes de valeurs.
3. On repère les graduations de profondeur (0, 1, 2 ... 16) -> axe profondeur <-> position y.
4. On détecte les traits roses/rouges qui séparent les couches dans la colonne
   "Lithologie" (analyse de l'image) -> limites des couches.
5. Chaque ligne d'essai (Pf, Pl, Em) reçoit une profondeur et la lithologie de sa couche.
"""
import re
import unicodedata
from io import BytesIO
 
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
 
try:
    import pytesseract
    from PIL import Image
 
    OCR_AVAILABLE = True
except Exception:  # OCR facultatif
    OCR_AVAILABLE = False
 
ZOOM = 3
NUM_RE = re.compile(r"^\d+(?:[.,]\d+)?$")
INT_RE = re.compile(r"^\d{1,2}$")
 
 
class PageSkip(Exception):
    """Page qui n'est pas une coupe pressiométrique exploitable."""
 
 
# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #
def cx(w):
    return (w[0] + w[2]) / 2
 
 
def cy(w):
    return (w[1] + w[3]) / 2
 
 
def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().strip()
 
 
def to_float(s: str) -> float:
    return float(s.replace(",", "."))
 
 
def cluster_1d(items, key, tol):
    items = sorted(items, key=key)
    groups = []
    for it in items:
        if groups and key(it) - key(groups[-1][-1]) <= tol:
            groups[-1].append(it)
        else:
            groups.append([it])
    return groups
 
 
def join_words(ws):
    if not ws:
        return ""
    h = float(np.median([w[3] - w[1] for w in ws])) or 1.0
    lines = cluster_1d(ws, cy, 0.6 * h)
    return " ".join(
        " ".join(w[4] for w in sorted(line, key=lambda w: w[0])) for line in lines
    ).strip()
 
 
# --------------------------------------------------------------------------- #
# Lecture des mots (texte natif, sinon OCR)
# --------------------------------------------------------------------------- #
def get_words(page):
    words = [
        (w[0], w[1], w[2], w[3], w[4].strip())
        for w in page.get_text("words")
        if w[4].strip()
    ]
    if len(words) >= 15:
        return words, "texte"
 
    if OCR_AVAILABLE:
        pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM), alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        try:
            d = pytesseract.image_to_data(
                img, lang="fra", config="--psm 11", output_type=pytesseract.Output.DICT
            )
        except Exception:
            d = pytesseract.image_to_data(
                img, config="--psm 11", output_type=pytesseract.Output.DICT
            )
        out = []
        for i, t in enumerate(d["text"]):
            t = t.strip()
            if not t:
                continue
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            out.append((x / ZOOM, y / ZOOM, (x + w) / ZOOM, (y + h) / ZOOM, t))
        return out, "ocr"
 
    return words, "texte"
 
 
def detect_name(page, words, fallback):
    texts = [page.get_text("text"), " ".join(w[4] for w in words)]
    for txt in texts:
        m = re.search(r"Sondage\s*:\s*([A-Za-z0-9_\-/.]+)", txt, flags=re.I)
        if m:
            return m.group(1).strip()
    for txt in texts:
        m = re.search(r"\b(S[A-Z]?[_\-][A-Za-z]+[_\-]?\d{1,4})\b", txt)
        if m:
            return m.group(1)
    return fallback
 
 
def format_name(name: str, compact: bool) -> str:
    """SP_Rem_001 -> SP_Rem001 si compact."""
    if compact:
        return re.sub(r"[_\-](\d+)$", r"\1", name)
    return name
 
 
# --------------------------------------------------------------------------- #
# Détection des limites de couches (traits roses/rouges dans la colonne lithologie)
# --------------------------------------------------------------------------- #
def detect_boundaries(page, x_lo, x_hi, y_lo, y_hi):
    pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    arr = arr[:, :, :3].astype(int)
 
    ya, yb = max(int(y_lo * ZOOM), 0), min(int(y_hi * ZOOM), arr.shape[0])
    xa, xb = max(int(x_lo * ZOOM), 0), min(int(x_hi * ZOOM), arr.shape[1])
    if yb <= ya or xb <= xa:
        return []
 
    sub = arr[ya:yb, xa:xb]
    r, g, b = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]
    mask = (r >= 180) & ((r - g) >= 40) & ((r - b) >= 40)
    frac = mask.mean(axis=1)
    rows = np.where(frac > 0.5)[0]
    if len(rows) == 0:
        return []
 
    groups = cluster_1d(list(rows), key=lambda v: v, tol=3)
    return [y_lo + float(np.mean(g_)) / ZOOM + (ya / ZOOM - y_lo) for g_ in groups]
 
 
# --------------------------------------------------------------------------- #
# Analyse d'une page
# --------------------------------------------------------------------------- #
def analyze_page(page, page_no, step=1.5, assign="axe", compact_names=True):
    words, src = get_words(page)
    pw = page.rect.width
 
    name = format_name(detect_name(page, words, f"page_{page_no}"), compact_names)
 
    # --- en-têtes des colonnes de mesures
    hdr = {}
    for key in ("pf", "pl", "em"):
        cands = [w for w in words if re.match(rf"^{key}\b", norm(w[4]))]
        if not cands:
            raise PageSkip("en-têtes Pf / Pl / Em introuvables")
        hdr[key] = min(cands, key=lambda w: w[1])
    hdr_bottom = max(w[3] for w in hdr.values())
 
    lith_hdr = next((w for w in words if norm(w[4]) == "lithologie"), None)
    cpf, cpl, cem = cx(hdr["pf"]), cx(hdr["pl"]), cx(hdr["em"])
    if not (cpf < cpl < cem):
        raise PageSkip("ordre des colonnes Pf / Pl / Em inattendu")
    lith_cx = cx(lith_hdr) if lith_hdr else cpf - 0.3 * pw
 
    # --- axe de profondeur (graduations entières dans la colonne de gauche)
    labels = [
        w
        for w in words
        if INT_RE.match(w[4]) and w[1] > hdr_bottom and cx(w) < lith_cx - 0.08 * pw
    ]
    uniq = {}
    for w in labels:
        uniq.setdefault(int(w[4]), w)
    if len(uniq) < 3:
        raise PageSkip("axe de profondeur introuvable")
    depths = np.array(sorted(uniq))
    ys = np.array([cy(uniq[d]) for d in depths])
    a, b = np.polyfit(depths, ys, 1)  # y = a*d + b
    if a <= 0:
        raise PageSkip("axe de profondeur incohérent")
    dmax = float(depths.max())
    labels_x1 = max(w[2] for w in uniq.values())
 
    # --- valeurs numériques par colonne
    half1, half2 = (cpl - cpf) / 2, (cem - cpl) / 2
    zones = {
        "Pf": (cpf - half1, (cpf + cpl) / 2),
        "Pl": ((cpf + cpl) / 2, (cpl + cem) / 2),
        "Em": ((cpl + cem) / 2, cem + half2 + 0.02 * pw),
    }
    y_row_lo = b - 0.15 * a
    y_row_hi = a * dmax + b + 0.35 * a
 
    pts = []
    for w in words:
        if not NUM_RE.match(w[4]):
            continue
        if not (y_row_lo <= cy(w) <= y_row_hi) or w[1] <= hdr_bottom:
            continue
        for col, (x0, x1) in zones.items():
            if x0 <= cx(w) < x1:
                pts.append((cy(w), col, to_float(w[4]), cx(w)))
 
    if not pts:
        raise PageSkip("aucune valeur Pf / Pl / Em détectée")
 
    rows = []
    for g in cluster_1d(pts, key=lambda p: p[0], tol=0.4 * a):
        row = {"y": float(np.mean([p[0] for p in g]))}
        for col in ("Pf", "Pl", "Em"):
            c = sorted([p for p in g if p[1] == col], key=lambda p: p[3])
            row[col] = c[0][2] if c else np.nan
        rows.append(row)
 
    # --- profondeur de chaque essai (grille de `step` mètres)
    est = [(r["y"] - b) / a for r in rows]
    ks = [max(1, int(round(e / step))) for e in est]
    if len(set(ks)) == len(ks) and ks == sorted(ks):
        depth_list = [k * step for k in ks]
    else:  # repli : on numérote dans l'ordre
        depth_list = [(i + 1) * step for i in range(len(rows))]
 
    # --- couches de lithologie
    y_lo = (lith_hdr[3] + 2) if lith_hdr else (b - 0.5 * a)
    y_hi = a * dmax + b + 0.2 * a
    x_text_lo = labels_x1 + 2
    x_text_hi = lith_cx + 0.11 * pw
 
    lith_words = [
        w
        for w in words
        if x_text_lo <= cx(w) <= x_text_hi
        and y_lo <= cy(w) <= y_hi
        and re.search(r"[A-Za-zÀ-ÿ]", w[4])
        and "neant" not in norm(w[4])
    ]
 
    bnds = detect_boundaries(page, lith_cx - 0.06 * pw, lith_cx + 0.06 * pw, y_lo, y_hi)
    method = "traits"
    if bnds:
        edges = [y_lo] + sorted(bnds) + [y_hi]
    else:
        method = "milieux"
        if lith_words:
            h = float(np.median([w[3] - w[1] for w in lith_words])) or 1.0
            grp = cluster_1d(lith_words, cy, 1.6 * h)
            centers = [float(np.mean([cy(w) for w in g])) for g in grp]
            mids = [(centers[i] + centers[i + 1]) / 2 for i in range(len(centers) - 1)]
            edges = [y_lo] + mids + [y_hi]
        else:
            edges = [y_lo, y_hi]
 
    layers = []
    for t, bt in zip(edges[:-1], edges[1:]):
        layers.append(join_words([w for w in lith_words if t <= cy(w) < bt]))
 
    # --- assemblage
    out = []
    for r, d in zip(rows, depth_list):
        ypos = (a * d + b) if assign == "axe" else r["y"]
        idx = sum(1 for e in edges[1:-1] if e < ypos)
        idx = min(idx, len(layers) - 1)
        out.append(
            {
                "Nom du sondage": name,
                "Profondeur (m)": d,
                "Lithologie": layers[idx],
                "Pf (MPa)": r["Pf"],
                "Pl* (MPa)": r["Pl"],
                "Em (MPa)": r["Em"],
            }
        )
 
    info = {
        "page": page_no,
        "sondage": name,
        "source": src,
        "essais": len(rows),
        "limites": method,
        "couches": layers,
    }
    return out, info
 
 
# --------------------------------------------------------------------------- #
# Traitement d'un PDF complet
# --------------------------------------------------------------------------- #
def process_pdf(pdf_bytes, step=1.5, assign="axe", compact_names=True, progress=None):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_rows, infos, skipped = [], [], []
    n = len(doc)
    for i in range(n):
        try:
            rows, info = analyze_page(doc[i], i + 1, step, assign, compact_names)
            all_rows.extend(rows)
            infos.append(info)
        except PageSkip as e:
            skipped.append((i + 1, str(e)))
        except Exception as e:  # une page défaillante ne bloque pas le reste
            skipped.append((i + 1, f"erreur : {e}"))
        if progress:
            progress((i + 1) / n)
    doc.close()
    return pd.DataFrame(all_rows), infos, skipped
 
 
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Pressiometre")
        ws = writer.sheets["Pressiometre"]
        for col_cells in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = min(width + 3, 40)
    return out.getvalue()
