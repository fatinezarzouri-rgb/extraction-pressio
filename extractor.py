"""
Extraction générale des coupes d'essais pressiométriques (PDF -> tableau).

Les coupes sont des IMAGES (pas de texte dans le PDF). Méthode :

1. Géométrie par analyse d'image (aucune coordonnée fixe) :
   - lignes verticales longues   -> colonnes Pf | Pl | Em et colonne Lithologie
   - lignes horizontales du cadre -> 0 m (haut) et bas du graphique -> axe de profondeur
   - traits roses               -> limites des couches (sinon : changement de symbole)
2. "Sondage : XX"  -> nom du sondage XX.
3. Chaque ligne de valeurs (Pf, Pl, Em sont au même niveau) : on projette son niveau
   (y) sur l'axe de profondeur -> profondeur de l'essai.
4. La profondeur est comparée aux zones de lithologie :
   une profondeur située exactement sur une limite appartient à la couche du dessus
   (ex. 0-0.5 terre végétale, 0.5-1.5 tuf calcaire : 1.5 m -> tuf calcaire).
5. Les chiffres sont relus un par un (agrandis) pour bien capter "." et ",".
"""
import difflib
import re
from collections import Counter
from io import BytesIO

import numpy as np
import pandas as pd

try:
    import fitz  # PyMuPDF
except Exception:  # permet de tester les fonctions sans PyMuPDF
    fitz = None

try:
    import pytesseract
    from PIL import Image

    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

ZOOM = 3
TEXT_BAND_FRAC = 0.41    # zone de texte de la lithologie (fraction de la largeur Pf..Em)
SYMBOL_BAND = (0.59, 0.43)  # zone des symboles (hachures), depuis la limite droite de la lithologie
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
    name = name.strip("._-/ ").replace("!", "1")
    m = re.match(r"^(.*?)([_\-]?)([0-9OolI|CQD]{1,4})$", name)
    if m:
        digits = m.group(3).translate(str.maketrans("OolI|CQD", "00111000"))
        name = m.group(1) + m.group(2) + digits
    if compact:
        name = re.sub(r"[_\-](\d+)$", r"\1", name)
    return name


def parse_value(text, col):
    """Lit un nombre ('.' ou ',' décimal). Si la virgule a été perdue par l'OCR, la restitue."""
    t = re.sub(r"[^0-9.,]", "", str(text)).replace(",", ".")
    if t.endswith("."):  # chiffre décimal coupé (ex. collé au cadre) : lecture non fiable
        return None
    t = t.strip(".")
    if not t:
        return None
    if t.count(".") > 1:
        head, *rest = t.split(".")
        t = head + "." + "".join(rest)
    if "." not in t and len(t) >= 2:
        # Em : 1 décimale (71.8, 1811.5) ; Pf/Pl : valeur < 10 avec 2 décimales (2.54)
        t = t[:-1] + "." + t[-1] if col == "Em" else t[0] + "." + t[1:]
    if not NUM_RE.match(t):
        return None
    v = float(t)
    if col in ("Pf", "Pl") and v >= 20:
        return None
    return v


# --------------------------------------------------------------------------- #
# Image
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
    if lines:  # le cadre = lignes les plus longues ; on écarte les courbes quasi verticales
        longest = max(l[2] - l[1] for l in lines)
        lines = [l for l in lines if (l[2] - l[1]) >= 0.8 * longest]
    return lines


def find_plot_frame(gray_ink, x0, x1):
    band = gray_ink[:, int(x0):int(x1)]
    rows = np.flatnonzero(band.mean(axis=1) > 0.85)
    if rows.size == 0:
        raise PageSkip("cadre horizontal du graphique introuvable")
    ys = [float(np.mean(g)) for g in cluster_1d(list(rows), key=lambda v: v, tol=3)]
    if len(ys) < 2:
        raise PageSkip("cadre horizontal du graphique introuvable")
    i = int(np.argmax(np.diff(ys)))
    return ys[i], ys[i + 1]


def detect_pink_lines(arr, x0, x1, top, bottom):
    """Traits roses/rouges pâles : profil de 'rosé' (R - moyenne(G,B)) moyenné sur la largeur."""
    sub = arr[int(top):int(bottom), int(x0):int(x1)].astype(np.float32)
    pk = np.clip(sub[:, :, 0] - (sub[:, :, 1] + sub[:, :, 2]) / 2, 0, None)
    prof = pk.mean(axis=1)
    rows = np.flatnonzero(prof > 4.0)
    if rows.size == 0:
        return []
    out = []
    for g_ in cluster_1d(list(rows), key=lambda v: v, tol=4):
        w = prof[g_]
        out.append(top + float(np.average(g_, weights=w)))
    return out


def detect_symbol_changes(dark, x0, x1, top, bottom, a):
    """Repli : limites = endroits où la texture des hachures change nettement."""
    reg = (dark[int(top):int(bottom), int(x0):int(x1)] < 150).astype(np.float32)
    if reg.shape[0] < 10 or reg.shape[1] < 5:
        return []
    dx = np.abs(np.diff(reg, axis=1)).mean(axis=1)[:-1]
    dy = np.abs(np.diff(reg, axis=0)).mean(axis=1)
    ink = reg.mean(axis=1)[:-1]
    feats = np.stack([ink, dx, dy], axis=1)
    win = max(int(0.5 * a), 4)
    n = feats.shape[0]
    score = np.zeros(n)
    for y in range(win, n - win):
        score[y] = np.abs(feats[y - win:y].mean(0) - feats[y:y + win].mean(0)).sum()
    thr = score.mean() + 3 * score.std()
    out = []
    for y in np.flatnonzero(score > thr):
        if out and y - out[-1][-1] <= 0.4 * a:
            out[-1].append(int(y))
        else:
            out.append([int(y)])
    return [top + float(np.mean(g)) for g in out]


# --------------------------------------------------------------------------- #
# Lecture de texte : texte natif (si présent), sinon OCR ciblé
# --------------------------------------------------------------------------- #
def native_words(page, rect):
    if page is None:
        return []
    r = fitz.Rect(*[v / ZOOM for v in rect])
    return [
        (w[0] * ZOOM, w[1] * ZOOM, w[2] * ZOOM, w[3] * ZOOM, w[4].strip())
        for w in page.get_text("words", clip=r)
        if w[4].strip()
    ]


def _tess(img, psm, whitelist=None, lang=None, data=True):
    cfg = f"--psm {psm}"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    kw = {"config": cfg}
    if lang:
        kw["lang"] = lang
    fn = pytesseract.image_to_data if data else pytesseract.image_to_string
    if data:
        kw["output_type"] = pytesseract.Output.DICT
    try:
        return fn(img, **kw)
    except Exception:
        kw.pop("lang", None)
        return fn(img, **kw)


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
        ex, half = ex if isinstance(ex, tuple) else (ex, 5)
        a, b = int(ex - x0 - half), int(ex - x0 + half + 1)
        bw[:, max(a, 0):max(b, 0)] = 255
    pad = 20
    img = Image.fromarray(np.pad(bw, pad, constant_values=255))
    d = _tess(img, psm, whitelist, lang)
    words = []
    for i, t in enumerate(d["text"]):
        t = str(t).strip()
        if t:
            l = d["left"][i] - pad + x0
            tp = d["top"][i] - pad + y0
            words.append((l, tp, l + d["width"][i], tp + d["height"][i], t))
    return words


def read_words(page, dark, rect, **kw):
    ws = native_words(page, rect)
    return ws if ws else ocr_crop(dark, rect, **kw)


def reread_numbers(dark, w):
    """Relit un nombre isolé avec plusieurs réglages (échelle / seuil) ; renvoie les lectures."""
    if not OCR_AVAILABLE:
        return []
    x0, y0, x1, y1 = [int(v) for v in w[:4]]
    m = 8
    crop = dark[max(y0 - m, 0):y1 + m, max(x0 - m, 0):x1 + m]
    if crop.size == 0:
        return []
    outs = []
    for scale, thr in ((1, 120), (1, 160)):
        c = crop
        if scale != 1:
            c = np.array(Image.fromarray(crop).resize(
                (crop.shape[1] * scale, crop.shape[0] * scale), Image.BICUBIC))
        bw = np.where(c < thr, 0, 255).astype(np.uint8)
        img = Image.fromarray(np.pad(bw, 30, constant_values=255))
        outs.append(_tess(img, 7, "0123456789.,", data=False).strip())
    return outs


NAME_TOKEN = re.compile(r"(S[Pp][_\-][A-Za-z0-9]{2,6}[_\-][0-9OolI!|CQD]{3})")


def read_sondage_name(page, dark, rect):
    """Renvoie XX dans 'Sondage : XX' (texte natif si présent, sinon OCR en plusieurs essais)."""
    if page is not None:
        m = re.search(r"Sondage\s*[:;.]?\s*([A-Za-z0-9_\-/]+)", page.get_text("text"), re.I)
        if m:
            return m.group(1)
    x0, y0, x1, y1 = [int(v) for v in rect]
    for scale, psm, thr in ((1, 6, 150), (1, 11, 150), (1, 6, 190), (2, 6, 170), (2, 11, 170), (2, 6, 200)):
        crop = dark[max(y0, 0):y1, max(x0, 0):x1]
        if crop.size == 0:
            return ""
        if scale != 1:
            crop = np.array(Image.fromarray(crop).resize(
                (crop.shape[1] * scale, crop.shape[0] * scale), Image.BICUBIC))
        ws = read_words(page, crop, (0, 0, crop.shape[1], crop.shape[0]), psm=psm, lang="fra", thr=thr) \
            if page is None else read_words(page, dark, rect, psm=psm, lang="fra", thr=thr)
        s = " ".join(w[4] for w in sorted(ws, key=lambda w: (round(cy(w) / 15), w[0])))
        m = re.search(r"Sondage\s*[:;.]?\s*([A-Za-z0-9_\-/]+)", s, flags=re.I)
        if m and NAME_TOKEN.fullmatch(m.group(1).replace("!", "1")):
            return m.group(1)
        m = NAME_TOKEN.search(s.replace(" ", ""))
        if m:
            return m.group(1)
    return ""


# --------------------------------------------------------------------------- #
# Analyse d'une page (tableau numpy RGB)
# --------------------------------------------------------------------------- #
def analyze_array(arr, page_no, page=None, step=0.5, assign="axe", compact_names=True):
    H, W, _ = arr.shape
    dark = arr.min(axis=2)
    ink = dark < 225
    ri, gi, bi = (arr[:, :, k].astype(np.int16) for k in range(3))
    pink = (ri >= 200) & ((ri - gi) >= 15) & ((ri - bi) >= 15)
    gray_ink = ink & ~pink

    # ---- colonnes
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

    # ---- axe de profondeur (0 m = haut du cadre)
    top, bottom = find_plot_frame(gray_ink, L[0] + 6, L[3] - 6)
    strip = (lith_hi - 0.80 * Wm, top - 4, lith_hi - 0.62 * Wm, bottom + 0.02 * H)
    lab = read_words(page, dark, strip, psm=11, whitelist="0123456789", thr=150)
    ests = [
        (cy(w) - top) / int(w[4])
        for w in lab
        if re.fullmatch(r"\d{1,2}", w[4]) and 4 <= int(w[4]) <= 40
    ]
    dmax = 16.0
    if len(ests) >= 3 and np.median(ests) > 0:
        c = round((bottom - top) / float(np.median(ests)))
        if 5 <= c <= 40:
            dmax = float(c)
    a = (bottom - top) / dmax  # pixels par mètre

    # ---- nom du sondage
    name_rect = (lith_hi - 0.10 * Wm, top - 0.16 * (bottom - top), L[3], top)
    raw = read_sondage_name(page, dark, name_rect)
    name = format_name(raw, compact_names) if raw else f"page_{page_no}"

    # ---- valeurs Pf / Pl / Em
    vrect = (L[0] + 5, top + 3, L[3] + 3, bottom - 3)
    vws = read_words(page, dark, vrect, psm=11, whitelist="0123456789.,", thr=140,
                     erase_cols=(L[1], L[2]))
    pts = []
    has_native = bool(native_words(page, vrect))
    for w in vws:
        x = cx(w)
        col = "Pf" if x < L[1] else ("Pl" if x < L[2] else "Em")
        texts = [w[4]] + ([] if has_native else reread_numbers(dark, w))
        votes = Counter(v for v in (parse_value(t, col) for t in texts) if v is not None)
        if votes:
            best = max(votes.values())
            first = next(v for v in (parse_value(t, col) for t in texts) if v in votes and votes[v] == best)
            pts.append((cy(w), col, first, x))
    if not pts:
        raise PageSkip("aucune valeur détectée")

    rows = []
    for grp in cluster_1d(pts, key=lambda p: p[0], tol=0.4 * a):
        row = {"y": float(np.mean([p[0] for p in grp]))}
        for col in ("Pf", "Pl", "Em"):
            c = sorted([p for p in grp if p[1] == col], key=lambda p: p[3])
            row[col] = c[0][2] if c else np.nan
        rows.append(row)

    # ---- profondeur de chaque essai = projection du niveau (y) sur l'axe
    depth_list = [max(step, round(((r_["y"] - top) / a) / step) * step) for r_ in rows]

    # ---- limites des couches : traits roses, sinon changement de symbole
    lo, hi = top + 0.1 * a, bottom - 0.1 * a
    bnds = [y for y in detect_pink_lines(arr, lith_hi - 0.50 * Wm, lith_hi - 0.05 * Wm, top, bottom)
            if lo < y < hi]
    how = "traits"
    if not bnds:
        how = "symboles"
        sx0, sx1 = lith_hi - SYMBOL_BAND[0] * Wm, lith_hi - SYMBOL_BAND[1] * Wm
        bnds = [y for y in detect_symbol_changes(dark, sx0, sx1, top, bottom, a) if lo < y < hi]
    bnds = sorted(bnds)
    edges = [top] + bnds + [bottom]
    bnd_depth = [round(((y - top) / a) / 0.5) * 0.5 for y in bnds]  # limites arrondies à 0.5 m

    layers = []
    for t, bt in zip(edges[:-1], edges[1:]):
        ws = read_words(page, dark, (text_lo, t + 2, lith_hi - 3, bt - 2), psm=6, lang="fra", thr=150)
        layers.append(clean_label(join_words(ws)))

    # ---- zone de chaque profondeur : une profondeur sur une limite -> couche du dessus
    out = []
    for r_, d in zip(rows, depth_list):
        if assign == "axe":
            idx = sum(1 for bd in bnd_depth if bd < d - 1e-6)
        else:
            idx = sum(1 for e in bnds if e < r_["y"])
        idx = min(idx, len(layers) - 1)
        out.append({
            "Nom du sondage": name,
            "Profondeur (m)": d,
            "Lithologie": layers[idx],
            "Pf (MPa)": r_["Pf"],
            "Pl* (MPa)": r_["Pl"],
            "Em (MPa)": r_["Em"],
        })

    zones = []
    prev = 0.0
    for bd, lab_ in zip(bnd_depth + [dmax], layers):
        zones.append(f"{prev:g}-{bd:g} {lab_}")
        prev = bd
    info = {
        "page": page_no,
        "sondage": name,
        "source": "texte" if native_words(page, vrect) else "ocr",
        "essais": len(rows),
        "limites": f"{len(bnds)} ({how})",
        "couches": zones,
    }
    return out, info


def analyze_page(page, page_no, step=0.5, assign="axe", compact_names=True):
    return analyze_array(render(page), page_no, page, step, assign, compact_names)


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


def snap_names(df, min_pages=3, cutoff=0.6):
    """Aligne les préfixes de noms mal lus par l'OCR (SP_Ree2 -> SP_Reta) sur les préfixes fréquents."""
    if df.empty:
        return df
    pat = re.compile(r"^(.*?)([_\-]?)(\d+)$")
    pref = {}
    for n in df["Nom du sondage"].unique():
        m = pat.match(n)
        if m:
            pref[n] = m.group(1)
    cnt = Counter(pref.values())
    frequent = [p for p, c in cnt.most_common() if c >= min_pages]

    def fix(n):
        m = pat.match(n)
        if not m or m.group(1) in frequent or not frequent:
            return n
        best = difflib.get_close_matches(m.group(1), frequent, n=1, cutoff=cutoff)
        return best[0] + m.group(2) + m.group(3) if best else n

    df = df.copy()
    df["Nom du sondage"] = df["Nom du sondage"].map(fix)
    return df


def process_pdf(pdf_bytes, step=0.5, assign="axe", compact_names=True, progress=None):
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
    df = snap_names(snap_labels(pd.DataFrame(all_rows)))
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
