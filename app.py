import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration de la page ---
st.set_page_config(
    page_title="Détection de Fraude Bancaire",
    page_icon="🏦",
    layout="wide"
)

# --- Chargement du modèle, du scaler, de l'encodeur et des données de performance ---
@st.cache_resource
def load_model():
    model = joblib.load("model/fraud_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    le_localisation = joblib.load("model/localisation_encoder.pkl")
    performance_data = joblib.load("model/performance_data.pkl")
    return model, scaler, le_localisation, performance_data

model, scaler, le_localisation, performance_data = load_model()

# --- Historique de session ---
if "historique" not in st.session_state:
    st.session_state.historique = []

# --- En-tête ---
st.title("🏦 Système de Détection de Fraude Bancaire")
st.markdown("Analysez une transaction ou un lot de transactions pour détecter un risque de fraude.")

# --- Menu latéral ---
mode = st.sidebar.radio(
    "Mode d'analyse",
    ["Transaction unique", "Fichier CSV (lot)", "Performance du modèle"]
)

# ============================
# MODE 1 : Transaction unique
# ============================
if mode == "Transaction unique":

    st.subheader("Saisie manuelle d'une transaction")

    col1, col2 = st.columns(2)

    with col1:
        montant = st.number_input("Montant de la transaction (FCFA)", min_value=0.0, value=50000.0)
        heure = st.slider("Heure de la transaction (0-23)", 0, 23, 12)

    with col2:
        localisation = st.selectbox(
            "Localisation de la transaction",
            le_localisation.classes_.tolist()
        )
        canal = st.selectbox("Canal", ["Mobile Money", "Carte", "Virement", "ATM"])

    if st.button("Analyser la transaction", type="primary"):

        # Encodage de la localisation avec le même encodeur qu'à l'entraînement
        localisation_encoded = le_localisation.transform([localisation])[0]

        features = np.array([[montant, heure, localisation_encoded]])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0][1]

        st.divider()

        if prediction == 1:
            st.error(f"⚠️ **Transaction suspecte** — Probabilité de fraude : {proba:.1%}")
        else:
            st.success(f"✅ **Transaction légitime** — Probabilité de fraude : {proba:.1%}")

        st.progress(float(proba))

        # Ajout à l'historique de la session
        st.session_state.historique.append({
            "Montant": montant,
            "Heure": heure,
            "Localisation": localisation,
            "Canal": canal,
            "Prédiction": "Fraude" if prediction == 1 else "Légitime",
            "Probabilité": f"{proba:.1%}"
        })

    if st.session_state.historique:
        st.divider()
        st.subheader("Historique de la session")
        st.dataframe(pd.DataFrame(st.session_state.historique))

# ============================
# MODE 2 : Fichier CSV (lot)
# ============================
elif mode == "Fichier CSV (lot)":

    st.subheader("Analyse par lot (fichier CSV)")

    fichier = st.file_uploader("Déposez un fichier CSV de transactions", type=["csv"])

    if fichier is not None:
        df = pd.read_csv(fichier)
        st.write("Aperçu des données :", df.head())

        if st.button("Lancer l'analyse du lot"):

            X_scaled = scaler.transform(df)

            df["prediction"] = model.predict(X_scaled)
            df["probabilite_fraude"] = model.predict_proba(X_scaled)[:, 1]

            nb_fraudes = df["prediction"].sum()
            st.warning(f"**{nb_fraudes}** transaction(s) suspecte(s) détectée(s) sur {len(df)}.")

            # Graphique de répartition des prédictions
            col1, col2 = st.columns([1, 2])
            with col1:
                fig, ax = plt.subplots()
                df["prediction"].value_counts().sort_index().plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"])
                ax.set_xticklabels(["Légitime", "Fraude"], rotation=0)
                ax.set_ylabel("Nombre de transactions")
                st.pyplot(fig)

            with col2:
                st.dataframe(
                    df.style.apply(
                        lambda row: ["background-color: #ffcccc" if row["prediction"] == 1 else "" for _ in row],
                        axis=1
                    )
                )

            csv_export = df.to_csv(index=False).encode("utf-8")
            st.download_button("Télécharger les résultats", csv_export, "resultats_analyse.csv", "text/csv")

# ============================
# MODE 3 : Performance du modèle
# ============================
else:

    st.subheader("Performance et explicabilité du modèle")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Importance des variables**")
        fig, ax = plt.subplots()
        ax.barh(performance_data["feature_names"], performance_data["feature_importances"], color="#3498db")
        ax.set_xlabel("Importance")
        st.pyplot(fig)

    with col2:
        st.markdown(f"**Courbe ROC** (AUC = {performance_data['auc']:.2f})")
        fig, ax = plt.subplots()
        ax.plot(performance_data["fpr"], performance_data["tpr"], label=f"AUC = {performance_data['auc']:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("Taux de faux positifs")
        ax.set_ylabel("Taux de vrais positifs")
        ax.legend()
        st.pyplot(fig)

    st.info("Ces indicateurs sont calculés une fois lors de l'entraînement (`train_model.py`) et rechargés ici.")

# --- Pied de page ---
st.sidebar.markdown("---")
st.sidebar.caption("Projet pédagogique — Détection de fraude bancaire par IA")