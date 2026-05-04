import streamlit as st
import pandas as pd
import fitz, re, cv2, pytesseract, numpy as np
from PIL import Image
from io import BytesIO

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.title("Extraction pressiométrique PDF → Excel")

uploaded = st.file_uploader("Importer le PDF", type=["pdf"])

HEADER_BOX = (0.18, 0.08, 0.99, 0.30)
LITHO_BOX  = (0.14, 0.28, 0.42, 0.90)
PL_BOX     = (0.62, 0.28, 0.78, 0.90)
EM_BOX     = (0.78, 0.28, 0.99, 0.90)

DEPTH_MAX = 15.0


def crop(img, box):
    w, h = img.size
    return img.crop((int(w*box[0]), int(h*box[1]), int(w*box[2]), int(h*box[3])))


def fr(v):
    if v == "" or v is None:
        return ""
    return str(v).replace(".", ",")


def to_float(v):
    try:
        return float(str(v).replace(",", "."))
    except:
        return None


def snap_depth(d):
    return round(round(d / 1.5) * 1.5, 1)


def color_bw(img, color):
    arr = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    if color == "blue":
        mask = cv2.inRange(hsv, np.array([90, 35, 35]), np.array([145, 255, 255]))
    else:
        m1 = cv2.inRange(hsv, np.array([0, 35, 35]), np.array([15, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([160, 35, 35]), np.array([180, 255, 255]))
        mask = m1 + m2

    out = np.ones(mask.shape, dtype=np.uint8) * 255
    out[mask > 0] = 0
    out = cv2.dilate(out, np.ones((2, 2), np.uint8), iterations=1)
    return Image.fromarray(out)


def ocr_text(img, psm=6):
    return pytesseract.image_to_string(img, lang="fra+eng", config=f"--psm {psm}")


def extract_header(img):
    text = ocr_text(crop(img, HEADER_BOX), 6)

    m = re.search(r"(SP\s*[_\-]?\s*[A-Za-z]+\s*[_\-]?\s*\d+)", text, re.I)
    sondage = ""
    if m:
        sondage = re.sub(r"\s+", "", m.group(1)).replace("-", "_")
        sondage = sondage.replace("_0", "0")

    coords = re.findall(r"\d{3}\s?\d{3}[.,]\d+", text)

    def fix(c):
        return c.replace(" ", "").replace(".", ",")

    x = fix(coords[0]) if len(coords) >= 1 else ""
    y = fix(coords[1]) if len(coords) >= 2 else ""

    return sondage, x, y


def extract_lithology(img):
    text = ocr_text(crop(img, LITHO_BOX), 6).lower()

    if "tuf" in text and "graveleux" in text:
        litho1 = "Tuf graveleux"
    elif "tuf" in text:
        litho1 = "Tuf calcaire"
    else:
        litho1 = ""

    if "calcaire" in text:
        litho2 = "Calcaire dure"
    elif "schiste" in text or "schiteuse" in text:
        litho2 = "Roche schisteuse dure"
    elif "granit" in text or "graniste" in text:
        litho2 = "Roche granitique grise"
    else:
        litho2 = litho1

    return litho1, litho2


def extract_pl_points(img):
    zone = crop(img, PL_BOX)
    bw = color_bw(zone, "blue")

    df = pytesseract.image_to_data(
        bw,
        lang="eng",
        config="--psm 6 -c tessedit_char_whitelist=0123456789.,",
        output_type=pytesseract.Output.DATAFRAME
    ).dropna(subset=["text"])

    page_w, page_h = img.size
    zone_y0 = int(page_h * PL_BOX[1])
    h = zone.size[1]

    points = []

    for _, r in df.iterrows():
        txt = str(r["text"]).strip()
        nums = re.findall(r"\d+[.,]\d+|\d+", txt)

        for n in nums:
            v = to_float(n)
            if v is None:
                continue

            if 0.1 <= v <= 30:
                y_local = r["top"] + r["height"] / 2
                y_global = zone_y0 + y_local
                depth = snap_depth((y_local / h) * DEPTH_MAX)

                if 0 < depth <= DEPTH_MAX:
                    points.append({
                        "depth": depth,
                        "pl": v,
                        "y_global": y_global
                    })

    final = {}
    for p in sorted(points, key=lambda x: x["depth"]):
        final[p["depth"]] = p

    return [final[d] for d in sorted(final.keys())]


def get_em_for_pl(img, y_global):
    page_w, page_h = img.size
    zone = crop(img, EM_BOX)

    em_y0 = int(page_h * EM_BOX[1])
    y_local = int(y_global - em_y0)

    w, h = zone.size

    band = zone.crop((
        0,
        max(0, y_local - 28),
        w,
        min(h, y_local + 28)
    ))

    bw = color_bw(band, "red")

    text = pytesseract.image_to_string(
        bw,
        lang="eng",
        config="--psm 7 -c tessedit_char_whitelist=0123456789.,"
    )

    nums = re.findall(r"\d+[.,]\d+|\d+", text)

    vals = []
    for n in nums:
        v = to_float(n)
        if v is None:
            continue

        # Em généralement > 10 MPa
        if 10 <= v <= 20000:
            vals.append(v)

    if not vals:
        return ""

    return max(vals)


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
        litho1, litho2 = extract_lithology(img)

        pl_points = extract_pl_points(img)

        for i, p in enumerate(pl_points):
            depth = p["depth"]
            em = get_em_for_pl(img, p["y_global"])

            rows.append({
                "Nom du sondages": sondage,
                "x": x if i == 0 else "",
                "y": y if i == 0 else "",
                "Profondeur (m)": fr(depth),
                "Lithologie": litho1 if depth <= 2.5 else litho2,
                "Pl* (MPa)": fr(round(p["pl"], 3)),
                "Em (MPa)": fr(round(em, 1)) if em != "" else ""
            })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Télécharger Excel",
        make_excel(df),
        "extraction_pressiometrique.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
