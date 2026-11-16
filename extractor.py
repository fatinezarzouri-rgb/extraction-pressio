
"""
Extraction générale des coupes d'essais pressiométriques (PDF -> tableau).
 
Les coupes sont des IMAGES (pas de texte dans le PDF). La méthode est donc :
1. Géométrie par analyse d'image (aucune coordonnée fixe) :
   - lignes verticales longues  -> colonnes Pf | Pl | Em et colonne Lithologie
   - lignes horizontales        -> haut (0 m) et bas du graphique -> axe de profondeur
   - traits roses               -> limites des couches de lithologie
2. OCR ciblé sur de petites zones (valeurs, nom du sondage, libellés de couches).
   Si le PDF contient du vrai texte dans ces zones, il est utilisé à la place de l'OCR.
3. Chaque ligne d'essai reçoit sa profondeur (grille de `step` m) et la lithologie
   de la couche correspondante.
"""
import difflib
import re
from collections import Counter
from io import BytesIO
 
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
 
try:
    import pytesseract
    from PIL import Image
 
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False
 
ZOOM = 3
TEXT_BAND_FRAC = 0.41  # largeur (en fraction de la zone de mesures) de la zone de texte lithologie
NUM_RE = re.compile(r"^\d+(?:\.\d+)?$")
 
 
class PageSkip(Exception):
    """Page qui n'est pas une coupe pressiométrique exploitable."""
 
 
# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #
def cx(w):
    return (w[0] + w[2]) / 2
 
 
def cy(w):
    return (w[1] + w[3]) / 2
 
 
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
 
 
def clean_label(s: str) -> str:
    s = re.sub(r"[^A-Za-zÀ-ÿ'’\- ]", " ", s)
    toks = [t for t in s.split() if len(t) > 1 or t.lower() in ("a", "à")]
    s = " ".join(toks)
    s = re.sub(r"(?<=\w) a (?=\w)", " à ", s)
    return s[:1].upper() + s[1:] if s else ""
 
 
def format_name(name: str, compact: bool) -> str:
    """SP_Rem_001 -> SP_Rem001 si compact."""
    name = name.strip("._-/ ")
    m = re.match(r"^(.*?)([_\-]?)([0-9OolI|]{1,4})$", name)
    if m:  # corrige O->0, l/I->1 dans la partie numérique
        digits = m.group(3).translate(str.maketrans("OolI|", "00111"))
        name = m.group(1) + m.group(2) + digits
    if compact:
        name = re.sub(r"[_\-](\d+)$", r"\1", name)
    return name
 
 
# --------------------------------------------------------------------------- #
# Image : rendu, lignes verticales / horizontales, traits roses
# --------------------------------------------------------------------------- #
def render(page):
    pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return arr[:, :, :3]
 
 
def longest_run(col, maxgap=6):
    idx = np.flatnonzero(col)
    if idx.size == 0:
        return 0, 0, 0
    breaks = np.where(np.diff(idx) > maxgap + 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [idx.size - 1]))
    lens = idx[ends] - idx[starts] + 1
    i = int(lens.argmax())
    return int(lens[i]), int(idx[starts[i]]), int(idx[ends[i]])
 
 
def find_vertical_lines(ink, min_len):
    colsum = ink.sum(axis=0)
    cand = []
    for x in np.flatnonzero(colsum >= min_len):
        n, y0, y1 = longest_run(ink[:, x])
        if n >= min_len:
            cand.append((int(x), y0, y1))
    lines = []
    for grp in cluster_1d(cand, key=lambda c: c[0], tol=4):
        best = max(grp, key=lambda c: c[2] - c[1])
        lines.append((float(np.mean([c[0] for c in grp])), best[1], best[2]))
    return lines
 
 
def find_plot_frame(gray_ink, x0, x1):
    band = gray_ink[:, int(x0):int(x1)]
    frac = band.mean(axis=1)
    rows = np.flatnonzero(frac > 0.85)
    if rows.size == 0:
        raise PageSkip("cadre horizontal du graphique introuvable")
    ys = [float(np.mean(g)) for g in cluster_1d(list(rows), key=lambda v: v, tol=3)]
    if len(ys) < 2:
        raise PageSkip("cadre horizontal du graphique introuvable")
    i = int(np.argmax(np.diff(ys)))
    return ys[i], ys[i + 1]
 
 
def detect_boundaries(arr, x0, x1, top, bottom):
    sub = arr[int(top):int(bottom), int(x0):int(x1)].astype(np.int16)
    r, g, b = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]
    mask = (r >= 200) & ((r - g) >= 15) & ((r - b) >= 15)
    frac = mask.mean(axis=1)
    rows = np.flatnonzero(frac > 0.4)
    if rows.size == 0:
        return []
    return [top + float(np.mean(g_)) for g_ in cluster_1d(list(rows), key=lambda v: v, tol=4)]
 
 
# --------------------------------------------------------------------------- #
# Lecture de texte : texte natif si présent, sinon OCR ciblé
# --------------------------------------------------------------------------- #
def native_words(page, rect):
    r = fitz.Rect(*[v / ZOOM for v in rect])
    return [
        (w[0] * ZOOM, w[1] * ZOOM, w[2] * ZOOM, w[3] * ZOOM, w[4].strip())
        for w in page.get_text("words", clip=r)
        if w[4].strip()
    ]
 
 
def ocr_crop(dark, rect, psm=6, whitelist=None, lang=None, thr=150, erase_cols=()):
    if not OCR_AVAILABLE:
        return []
    x0, y0, x1, y1 = [int(v) for v in rect]
    x0, y0 = max(x0, 0), max(y0, 0)
    crop = dark[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    bw = np.where(crop < thr, 0, 255).astype(np.uint8)
    for ex in erase_cols:
        a, b = int(ex - x0 - 5), int(ex - x0 + 6)
        bw[:, max(a, 0):max(b, 0)] = 255
    pad = 20
    bw = np.pad(bw, pad, constant_values=255)
    img = Image.fromarray(bw)
 
    cfg = f"--psm {psm}"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    kw = {"config": cfg, "output_type": pytesseract.Output.DICT}
    if lang:
        kw["lang"] = lang
    try:
        d = pytesseract.image_to_data(img, **kw)
    except Exception:
        kw.pop("lang", None)
        d = pytesseract.image_to_data(img, **kw)
 
    words = []
    for i, t in enumerate(d["text"]):
        t = str(t).strip()
        if not t:
            continue
        l = d["left"][i] - pad + x0
        tp = d["top"][i] - pad + y0
        words.append((l, tp, l + d["width"][i], tp + d["height"][i], t))
    return words
 
 
def read_words(page, dark, rect, **ocr_kw):
    ws = native_words(page, rect)
    return ws if ws else ocr_crop(dark, rect, **ocr_kw)
 
 
def read_sondage_name(page, dark, rect):
    txt = page.get_text("text")
    pats = [r"Sondage\s*[:;.]?\s*([A-Za-z0-9_\-/]+)"]
    for src in [txt]:
        for p in pats:
            m = re.search(p, src, flags=re.I)
            if m:
                return m.group(1)
    for psm, thr in ((6, 150), (11, 150), (6, 185), (11, 185)):
        ws = read_words(page, dark, rect, psm=psm, lang="fra", thr=thr)
        s = " ".join(w[4] for w in sorted(ws, key=lambda w: (round(cy(w) / 15), w[0])))
        m = re.search(r"Sondage\s*[:;.]?\s*([A-Za-z0-9_\-/]+)", s, flags=re.I)
        if m:
            return m.group(1)
        m = re.search(r"\b(S[A-Za-z]?[_\-][A-Za-z]{2,}[_\-]?[0-9OolI]{1,4})\b", s)
        if m:
            return m.group(1)
    return ""
 
 
# --------------------------------------------------------------------------- #
# Analyse d'une page
# --------------------------------------------------------------------------- #
def analyze_page(page, page_no, step=1.5, assign="axe", compact_names=True):
    arr = render(page)
    H, W, _ = arr.shape
    dark = arr.min(axis=2)
    ink = dark < 225
    ri, gi, bi = (arr[:, :, k].astype(np.int16) for k in range(3))
    pink = (ri >= 200) & ((ri - gi) >= 15) & ((ri - bi) >= 15)
    gray_ink = ink & ~pink
 
    # --- colonnes (lignes verticales longues)
    lines = find_vertical_lines(ink, int(0.30 * H))
    if len(lines) < 4:
        raise PageSkip("colonnes de mesures introuvables (pas une coupe ?)")
    xs = [l[0] for l in lines]
    L = xs[-4:]
    widths = np.diff(L)
    if widths.min() <= 0 or widths.max() / widths.min() > 2.5:
        raise PageSkip("colonnes Pf / Pl / Em incohérentes")
    Wm = L[3] - L[0]
 
    lith_hi = L[0] - 0.285 * Wm
    if len(xs) >= 6 and abs((L[0] - xs[-6]) - 0.285 * Wm) < 0.08 * Wm:
        lith_hi = xs[-6]
    text_lo = lith_hi - TEXT_BAND_FRAC * Wm
 
    # --- cadre horizontal -> axe de profondeur
    top, bottom = find_plot_frame(gray_ink, L[0] + 6, L[3] - 6)
 
    # --- nombre de mètres de l'axe (via les graduations, sinon 16)
    strip = (lith_hi - 0.78 * Wm, top - 4, lith_hi - 0.60 * Wm, bottom + 0.02 * H)
    lab = read_words(page, dark, strip, psm=11, whitelist="0123456789", thr=150)
    ests = []
    for w in lab:
        if re.fullmatch(r"\d{1,2}", w[4]) and 4 <= int(w[4]) <= 40:
            ests.append((cy(w) - top) / int(w[4]))
    dmax = 16.0
    if len(ests) >= 3 and np.median(ests) > 0:
        cand = round((bottom - top) / float(np.median(ests)))
        if 5 <= cand <= 40:
            dmax = float(cand)
    a = (bottom - top) / dmax  # pixels par mètre
 
    # --- nom du sondage
    chart_top = lines[-1][1]
    name_rect = (lith_hi - 0.10 * Wm, chart_top, L[3], top)
    raw_name = read_sondage_name(page, dark, name_rect)
    name = format_name(raw_name, compact_names) if raw_name else f"page_{page_no}"
 
    # --- valeurs Pf / Pl / Em
    vrect = (L[0] + 5, top + 3, L[3] - 5, bottom - 3)
    vws = read_words(
        page, dark, vrect, psm=11, whitelist="0123456789.", thr=100, erase_cols=(L[1], L[2])
    )
    pts = []
    for w in vws:
        t = w[4].strip(".")
        if not NUM_RE.match(t):
            continue
        x = cx(w)
        col = "Pf" if x < L[1] else ("Pl" if x < L[2] else "Em")
        pts.append((cy(w), col, float(t), x))
    if not pts:
        raise PageSkip("aucune valeur détectée")
 
    rows = []
    for grp in cluster_1d(pts, key=lambda p: p[0], tol=0.4 * a):
        row = {"y": float(np.mean([p[0] for p in grp]))}
        for col in ("Pf", "Pl", "Em"):
            c = sorted([p for p in grp if p[1] == col], key=lambda p: p[3])
            row[col] = c[0][2] if c else np.nan
        rows.append(row)
 
    # --- profondeur de chaque essai (grille de `step` m)
    ks = [max(1, int(round(((r_["y"] - top) / a) / step))) for r_ in rows]
    if len(set(ks)) == len(ks) and ks == sorted(ks):
        depth_list = [k * step for k in ks]
    else:
        depth_list = [(i + 1) * step for i in range(len(rows))]
 
    # --- couches de lithologie
    bnds = [
        y
        for y in detect_boundaries(arr, lith_hi - 0.50 * Wm, lith_hi - 0.05 * Wm, top, bottom)
        if top + 0.1 * a < y < bottom - 0.1 * a
    ]
    edges = [top] + sorted(bnds) + [bottom]
    layers = []
    for t, bt in zip(edges[:-1], edges[1:]):
        rect = (text_lo, t + 2, lith_hi - 3, bt - 2)
        ws = read_words(page, dark, rect, psm=6, lang="fra", thr=150)
        layers.append(clean_label(join_words(ws)))
 
    # --- assemblage
    out = []
    for r_, d in zip(rows, depth_list):
        ypos = (top + a * d) if assign == "axe" else r_["y"]
        idx = min(sum(1 for e in edges[1:-1] if e < ypos), len(layers) - 1)
        out.append(
            {
                "Nom du sondage": name,
                "Profondeur (m)": d,
                "Lithologie": layers[idx],
                "Pf (MPa)": r_["Pf"],
                "Pl* (MPa)": r_["Pl"],
                "Em (MPa)": r_["Em"],
            }
        )
 
    info = {
        "page": page_no,
        "sondage": name,
        "source": "ocr" if not native_words(page, vrect) else "texte",
        "essais": len(rows),
        "limites": f"{len(bnds)} trait(s)",
        "couches": layers,
    }
    return out, info
 
 
# --------------------------------------------------------------------------- #
# PDF complet
# --------------------------------------------------------------------------- #
def snap_labels(df, min_count=20, cutoff=0.8):
    """Corrige les petites erreurs d'OCR en alignant sur les libellés fréquents."""
    if df.empty:
        return df
    cnt = Counter(df.loc[df["Lithologie"] != "", "Lithologie"])
    frequent = [l for l, c in cnt.most_common() if c >= min_count]
 
    def fix(l):
        if not l or l in frequent:
            return l
        m = difflib.get_close_matches(l, frequent, n=1, cutoff=cutoff)
        return m[0] if m else l
 
    df = df.copy()
    df["Lithologie"] = df["Lithologie"].map(fix)
    return df
 
 
def process_pdf(pdf_bytes, step=1.5, assign="axe", compact_names=True, progress=None):
    if not OCR_AVAILABLE:
        raise RuntimeError("pytesseract / Pillow non installés (voir requirements.txt et packages.txt).")
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
        except Exception as e:
            skipped.append((i + 1, f"erreur : {e}"))
        if progress:
            progress((i + 1) / n)
    doc.close()
    df = pd.DataFrame(all_rows)
    df = snap_labels(df)
    return df, infos, skipped
 
 
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Pressiometre")
        ws = writer.sheets["Pressiometre"]
        for col_cells in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = min(width + 3, 40)
    return out.getvalue()
