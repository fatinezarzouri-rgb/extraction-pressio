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
