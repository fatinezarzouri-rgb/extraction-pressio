import pandas as pd
import streamlit as st
 
from extractor import process_pdf, to_excel_bytes
 
st.set_page_config(page_title="Coupes pressiométriques → Excel", layout="wide")
st.title("Coupes pressiométriques → Excel")
st.caption("Charge un ou plusieurs PDF de coupes (Pf, Pl, Em + lithologie) et récupère un tableau Excel.")
 
with st.sidebar:
    st.header("Paramètres")
    step = st.number_input("Pas entre les essais (m)", min_value=0.5, max_value=5.0, value=1.5, step=0.5)
    assign = st.selectbox(
        "Affectation de la lithologie",
        options=["axe", "texte"],
        format_func=lambda v: {
            "axe": "Selon la profondeur de l'essai (axe)",
            "texte": "Selon la position des valeurs sur la coupe",
        }[v],
        help="Si une lithologie te paraît décalée près d'une limite de couche, essaie l'autre mode.",
    )
    compact = st.checkbox("Nom court : SP_Rem001 (au lieu de SP_Rem_001)", value=True)
    with_pf = st.checkbox("Ajouter la colonne Pf (MPa)", value=False)
    debug = st.checkbox("Afficher le détail par page", value=False)
 
files = st.file_uploader("PDF des coupes", type=["pdf"], accept_multiple_files=True)
 
if st.button("Lancer l'extraction", type="primary"):
    if not files:
        st.error("Charge d'abord au moins un PDF.")
        st.stop()
 
    frames, all_infos, all_skipped = [], [], []
    for f in files:
        bar = st.progress(0.0, text=f"Lecture de {f.name}…")
        try:
            df, infos, skipped = process_pdf(
                f.read(),
                step=step,
                assign=assign,
                compact_names=compact,
                progress=lambda p, bar=bar, n=f.name: bar.progress(p, text=f"Lecture de {n}…"),
            )
        except Exception as e:
            st.exception(e)
            continue
        bar.empty()
        if not df.empty:
            df.insert(0, "Fichier", f.name)
            frames.append(df)
        all_infos += [(f.name, i) for i in infos]
        all_skipped += [(f.name, p, m) for p, m in skipped]
 
    if not frames:
        st.warning("Aucune donnée détectée.")
    else:
        result = pd.concat(frames, ignore_index=True)
        cols = ["Nom du sondage", "Profondeur (m)", "Lithologie"]
        if with_pf:
            cols.append("Pf (MPa)")
        cols += ["Pl* (MPa)", "Em (MPa)"]
        result = result[["Fichier"] + cols] if len(files) > 1 else result[cols]
 
        st.success(f"{result['Nom du sondage'].nunique()} sondages – {len(result)} lignes")
        st.dataframe(result, use_container_width=True, height=500)
 
        empty_lith = int((result["Lithologie"].astype(str).str.strip() == "").sum())
        missing_val = int(result[["Pl* (MPa)", "Em (MPa)"]].isna().any(axis=1).sum())
        if empty_lith:
            st.warning(f"{empty_lith} ligne(s) sans lithologie détectée – à vérifier.")
        if missing_val:
            st.warning(f"{missing_val} ligne(s) avec Pl ou Em manquant – à vérifier.")
 
        st.download_button(
            "Télécharger l'Excel",
            data=to_excel_bytes(result),
            file_name="pressiometre.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
 
    if all_skipped:
        with st.expander(f"{len(all_skipped)} page(s) ignorée(s)"):
            st.dataframe(pd.DataFrame(all_skipped, columns=["Fichier", "Page", "Raison"]))
 
    if debug and all_infos:
        with st.expander("Détail par page", expanded=True):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Fichier": fn,
                            "Page": i["page"],
                            "Sondage": i["sondage"],
                            "Lecture": i["source"],
                            "Essais": i["essais"],
                            "Limites": i["limites"],
                            "Couches": " | ".join(i["couches"]),
                        }
                        for fn, i in all_infos
                    ]
                ),
                use_container_width=True,
            )
 
