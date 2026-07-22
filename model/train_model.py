import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

# 1. Chargement avec le bon séparateur
df = pd.read_csv("data/Bank_transaction_scenario1.csv", sep=";")

# Nettoyage des noms de colonnes
df.columns = df.columns.str.strip()

# 2. Encodage de la variable cible Target (Texte -> Nombre)
df['Target'] = df['Target'].apply(lambda x: 1 if str(x).strip() in ['Fraude', 'Suspect'] else 0)

# 3. Construction des 3 features utilisées par l'application Streamlit
df["Date"] = pd.to_datetime(df["Date"])
df["Heure"] = df["Date"].dt.hour

le_localisation = LabelEncoder()
df["Localisation_encoded"] = le_localisation.fit_transform(df["Localisation"])

X = df[["Montant", "Heure", "Localisation_encoded"]]
y = df["Target"]

# 4. Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Entraînement
model = RandomForestClassifier(
    n_estimators=200, max_depth=10, class_weight="balanced", random_state=42
)
model.fit(X_train, y_train)

# 7. Évaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 8. Sauvegarde du modèle, du scaler ET de l'encodeur de localisation
joblib.dump(model, "model/fraud_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(le_localisation, "model/localisation_encoder.pkl")
print("Modèle, Scaler et Encodeur entraînés et sauvegardés avec succès !")