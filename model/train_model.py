import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 1. Chargement avec le bon séparateur
df = pd.read_csv("data/Bank_transaction_scenario1.csv", sep=";")

# Nettoyage des noms de colonnes
df.columns = df.columns.str.strip()

# 2. Suppression des colonnes non pertinentes pour le ML (ex: ID, Date si présente)
# On retire toute colonne d'identifiant ou de date brute qui contient du texte unique
colonnes_a_exclure = [col for col in df.columns if 'id' in col.lower() or 'date' in col.lower() or 'ref' in col.lower() or 'trans' in col.lower()]
df = df.drop(columns=[col for col in colonnes_a_exclure if col in df.columns and col != 'Target'], errors='ignore')

# 3. Encodage de la variable cible Target (Texte -> Nombre)
df['Target'] = df['Target'].apply(lambda x: 1 if str(x).strip() in ['Fraude', 'Suspect'] else 0)

# 4. Séparation features / cible
X = df.drop("Target", axis=1)
y = df["Target"]

# 5. Encodage des colonnes catégorielles restantes (ex: Status operation, Localisation)
X = pd.get_dummies(X, drop_first=True)

# 6. Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Entraînement
model = RandomForestClassifier(
    n_estimators=200, max_depth=10, class_weight="balanced", random_state=42
)
model.fit(X_train, y_train)

# 9. Évaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 10. Sauvegarde du modèle ET du scaler
joblib.dump(model, "model/fraud_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print(" Modèle et Scaler entraînés et sauvegardés avec succès !")