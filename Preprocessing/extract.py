import pandas as pd

# Charger le fichier CSV complet
df = pd.read_csv("murex_data.csv")

# Extraire les 1000 premières lignes
sample_df = df.head(1000)

# Enregistrer dans un nouveau fichier
sample_df.to_csv("murex_data_sample.csv", index=False)

print("✅ Échantillon extrait avec succès dans 'murex_data_sample.csv'")
