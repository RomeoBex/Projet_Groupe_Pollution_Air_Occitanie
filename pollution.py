import pandas as pd

# Remplacez 'votre_fichier.csv' par le chemin de votre fichier CSV
chemin_du_fichier = 'ind_occitanie_2020.csv'

# Chargez le fichier CSV dans un DataFrame en spécifiant l'encodage et le délimiteur
dataframe = pd.read_csv(chemin_du_fichier, delimiter=';', encoding='ISO-8859-1')

# Affichez les premières lignes du DataFrame pour vérification
print(dataframe.head())
