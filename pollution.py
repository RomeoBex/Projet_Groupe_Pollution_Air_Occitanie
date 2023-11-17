#%%

import pandas as pd
import matplotlib.pyplot as plt 

# Remplacez 'votre_fichier.csv' par le chemin de votre fichier CSV
chemin_du_fichier = 'ind_occitanie_2020.csv'

# Chargez le fichier CSV dans un DataFrame en spécifiant l'encodage et le délimiteur
dataframe = pd.read_csv(chemin_du_fichier, delimiter=';', encoding='ISO-8859-1')

# Supprimez les lignes contenant des valeurs NaN
dataframe = dataframe.dropna()

# Triez le DataFrame par ordre alphabétique sur la colonne 'lib_zone'
dataframe = dataframe.sort_values(by='lib_zone')

# Réinitialisez l'index après le tri
dataframe = dataframe.reset_index(drop=True)

# Affichez les premières lignes du DataFrame après le tri
print(dataframe.head())

# Affichez les premières lignes du DataFrame après suppression des lignes avec NaN
print(dataframe.head())




#%%
import matplotlib.pyplot as plt

# Colonnes des polluants
polluants = ['val_no2', 'val_so2', 'val_o3', 'val_pm10', 'val_pm25']

# Calcul de la somme des valeurs pour chaque polluant
somme_par_polluant = ca_ales_data[polluants].sum()

# Créez un diagramme à barres pour la somme de polluants
plt.figure(figsize=(12, 8))

# Utilisez la fonction bar pour créer un diagramme à barres
plt.bar(polluants, somme_par_polluant)

plt.title('Quantité de polluant par polluant pour CA Alès')
plt.xlabel('Polluants')
plt.ylabel('Somme des valeurs')
plt.xticks(rotation=45, ha='right')  # Rotation des étiquettes sur l'axe x pour une meilleure lisibilité
plt.show()

chemin_enregistrement = os.path.join(os.path.expanduser('~'), 'Desktop', 'diagramme_polluants.svg')

# Enregistrez le diagramme en tant que fichier SVG
plt.savefig(chemin_enregistrement, format='svg')

# Affichez le diagramme
plt.show()

# %%

#Carte interactive 
from ipyleaflet import Map, TileLayer, basemaps

# Remplacez l'URL par le service WMTS que vous souhaitez utiliser
wmts_url = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/WMTS"

# Créer une carte avec le service WMTS
carte = Map(center=(43.611015, 3.876733), zoom=12)

# Ajouter une couche WMTS à la carte
wmts_layer = TileLayer(url=wmts_url, name="WMTS Layer")
carte.add_layer(wmts_layer)

# Afficher la carte
carte

# %%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Remplacez 'votre_fichier.csv' par le chemin vers votre fichier CSV
chemin_fichier_csv = 'Mesure_mensuelle_Region_Occitanie_Polluants_Principaux.csv'

# Charger le fichier CSV dans un DataFrame
df = pd.read_csv(chemin_fichier_csv)

# Créer un diagramme à barres pour la variable 'nom_dept'
plt.figure(figsize=(12, 6))
sns.countplot(x='nom_dept', data=df)
plt.title('Répartition des données par département')
plt.xlabel('Département')
plt.ylabel('Nombre d\'occurrences')
plt.xticks(rotation=45, ha='right')  # Rotation des étiquettes sur l'axe x pour une meilleure lisibilité
plt.show()
