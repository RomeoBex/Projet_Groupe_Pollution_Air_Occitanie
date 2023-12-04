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

dossier_diapositives = "diapositives"
chemin_fichier_svg = os.path.join(dossier_diapositives, 'graph2.svg')
plt.savefig(chemin_fichier_svg, format='svg')

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
#fonctionne 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

dossier_diapositives = "diapositives"
chemin_fichier_svg = os.path.join(dossier_diapositives, 'graph1.svg')
plt.savefig(chemin_fichier_svg, format='svg')

plt.show()

#%%

#fonctionne mais pas ouf 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Remplacez 'votre_fichier.csv' par le chemin vers votre fichier CSV
chemin_fichier_csv = 'Mesure_mensuelle_Region_Occitanie_Polluants_Principaux.csv'

# Charger le fichier CSV dans un DataFrame
df = pd.read_csv(chemin_fichier_csv)

# Créer un nuage de points pour les coordonnées X et Y
plt.figure(figsize=(10, 6))
sns.scatterplot(x='X', y='Y', data=df, hue='nom_dept', palette='viridis', s=50)
plt.title('Nuage de points : Coordonnées X et Y par département')
plt.xlabel('Coordonnée X')
plt.ylabel('Coordonnée Y')
plt.legend(title='Département', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# %%

#fonctionne camembert dégradé de couleur 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os 

# Remplacez 'votre_fichier.csv' par le chemin vers votre fichier CSV
chemin_fichier_csv = 'Mesure_mensuelle_Region_Occitanie_Polluants_Principaux.csv'

# Charger le fichier CSV dans un DataFrame
df = pd.read_csv(chemin_fichier_csv)

# Créer un dégradé de couleurs pour la palette
couleurs = sns.color_palette("coolwarm", as_cmap=True)

# Créer un diagramme circulaire (camembert) pour la variable 'nom_dept'
plt.figure(figsize=(10, 8))
sns.set(style="whitegrid")  # Style de fond pour une meilleure lisibilité

# Tracer le camembert avec le dégradé de couleurs
df['nom_dept'].value_counts().plot.pie(autopct='%1.1f%%', startangle=140, cmap=couleurs)

plt.title('Répartition des données par département')
plt.axis('equal')  # Assure que le camembert est dessiné comme un cercle
plt.ylabel('')  # Supprimer l'étiquette de l'axe y pour plus de clarté

dossier_diapositives = "diapositives"
chemin_fichier_svg = os.path.join(dossier_diapositives, 'graph2.svg')
plt.savefig(chemin_fichier_svg, format='svg')

plt.show()


# %%

#fonctionne mais incompréhensible 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Remplacez 'votre_fichier.csv' par le chemin vers votre fichier CSV
chemin_fichier_csv = 'Mesure_mensuelle_Region_Occitanie_Polluants_Principaux.csv'

# Charger le fichier CSV dans un DataFrame
df = pd.read_csv(chemin_fichier_csv)

# Supprimer les valeurs NaN pour l'histogramme
valeurs_non_nan = df['valeur'].dropna()

# Créer un histogramme
plt.figure(figsize=(10, 6))
sns.histplot(valeurs_non_nan, bins=20, kde=True, color='skyblue')
plt.title('Distribution des valeurs de concentration')
plt.xlabel('Concentration')
plt.ylabel('Nombre d\'occurrences')
plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le fichier CSV dans un DataFrame
chemin_fichier_csv = 'Mesure_mensuelle_Region_Occitanie_Polluants_Principaux.csv'
df = pd.read_csv(chemin_fichier_csv)

# Créer une figure avec plusieurs sous-plots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

# Afficher les 5 premières lignes du DataFrame
print("5 premières lignes du DataFrame :")
print(df.head())

# Résumé statistique des variables numériques
summary_stats = df.describe()
print("\nRésumé statistique :")
print(summary_stats)

# Histogramme des valeurs de concentration
axes[0, 0].hist(df['valeur'].dropna(), bins=20, color='skyblue')
axes[0, 0].set_title('Distribution des valeurs de concentration')
axes[0, 0].set_xlabel('Concentration')
axes[0, 0].set_ylabel('Nombre d\'occurrences')

# Diagramme à barres pour la répartition par département
sns.countplot(x='nom_dept', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Répartition des données par département')
axes[0, 1].set_xlabel('Département')
axes[0, 1].set_ylabel('Nombre d\'occurrences')
axes[0, 1].tick_params(axis='x', rotation=45)

# Nuage de points pour les coordonnées X et Y
sns.scatterplot(x='X', y='Y', data=df, hue='nom_dept', palette='viridis', s=50, ax=axes[1, 0])
axes[1, 0].set_title('Nuage de points : Coordonnées X et Y par département')
axes[1, 0].set_xlabel('Coordonnée X')
axes[1, 0].set_ylabel('Coordonnée Y')
axes[1, 0].legend(title='Département', bbox_to_anchor=(1.05, 1), loc='upper left')

# Graphique à secteurs pour la répartition par typologie
typologie_counts = df['typologie'].value_counts()
axes[1, 1].pie(typologie_counts, labels=typologie_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
axes[1, 1].set_title('Répartition des stations par typologie')

# Ajuster l'espacement entre les sous-plots
plt.tight_layout()

# Afficher les graphiques
plt.show()


# %%
#pas le bon
import pandas as pd
from ipyleaflet import Map, TileLayer, basemaps, GeoJSON

# Charger le fichier CSV dans un DataFrame
chemin_fichier_csv = 'Mesure_mensuelle_Region_Occitanie_Polluants_Principaux.csv'
df = pd.read_csv(chemin_fichier_csv)

# Créer une colonne 'geometry' avec les coordonnées X et Y sous forme de GeoJSON
df['geometry'] = df.apply(lambda row: {"type": "Point", "coordinates": [row['X'], row['Y']]}, axis=1)

# Créer une carte avec le service WMTS
carte = Map(center=(43.611015, 3.876733), zoom=9)

# Ajouter une couche WMTS à la carte
wmts_url = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/WMTS"
wmts_layer = TileLayer(url=wmts_url, name="WMTS Layer")
carte.add_layer(wmts_layer)

# Créer une GeoJSON FeatureCollection à partir des données de votre DataFrame
geojson_data = {
    "type": "FeatureCollection",
    "features": []
}

for index, row in df.iterrows():
    feature = {
        "type": "Feature",
        "geometry": row['geometry'],
        "properties": {"nom_dept": row['nom_dept'], "valeur": row['valeur']}
    }
    geojson_data['features'].append(feature)

# Créer une couche GeoJSON pour les zones polluées
geojson_layer = GeoJSON(data=geojson_data, style={'color': 'red', 'opacity': 0.8, 'weight': 1.5})
carte.add_layer(geojson_layer)

# Afficher la carte
carte


#pour calculer la valeur seuil 
# Calculer la moyenne et l'écart type des valeurs de pollution
moyenne_pollution = df['valeur'].mean()
ecart_type_pollution = df['valeur'].std()

# Définir le seuil comme la moyenne plus 2 fois l'écart type
seuil_valeur_elevee = moyenne_pollution + 2 * ecart_type_pollution

print("Moyenne de la pollution:", moyenne_pollution)
print("Écart type de la pollution:", ecart_type_pollution)
print("Seuil de valeur élevée:", seuil_valeur_elevee)




# %%

#calcule de la valeur seuil 
# Calculer la moyenne et l'écart type des valeurs de pollution
moyenne_pollution = df['valeur'].mean()
ecart_type_pollution = df['valeur'].std()

# Définir le seuil comme la moyenne plus 2 fois l'écart type
seuil_valeur_elevee = moyenne_pollution + 2 * ecart_type_pollution

print("Moyenne de la pollution:", moyenne_pollution)
print("Écart type de la pollution:", ecart_type_pollution)
print("Seuil de valeur élevée:", seuil_valeur_elevee)


import pandas as pd
from ipyleaflet import Map, TileLayer, GeoJSON, Marker
import ipywidgets as widgets
from IPython.display import display

# Charger le fichier CSV dans un DataFrame
chemin_fichier_csv = 'Mesure_mensuelle_Region_Occitanie_Polluants_Principaux.csv'
df = pd.read_csv(chemin_fichier_csv)

# Créer une colonne 'geometry' avec les coordonnées X et Y sous forme de GeoJSON
df['geometry'] = df.apply(lambda row: {"type": "Point", "coordinates": [row['X'], row['Y']]}, axis=1)

# Valeur seuil
SEUIL_DE_VALEUR_ELEVEE = 23  # Définissez votre seuil ici

# Créer une carte avec le service WMTS
carte = Map(center=(43.611015, 3.876733), zoom=9)

# Ajouter une couche WMTS à la carte
wmts_url = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/WMTS"
wmts_layer = TileLayer(url=wmts_url, name="WMTS Layer")
carte.add_layer(wmts_layer)

# Créer une GeoJSON FeatureCollection à partir des données de votre DataFrame
geojson_data = {
    "type": "FeatureCollection",
    "features": []
}

# Marqueurs pour les valeurs élevées
high_value_markers = []

for index, row in df.iterrows():
    feature = {
        "type": "Feature",
        "geometry": row['geometry'],
        "properties": {"nom_dept": row['nom_dept'], "valeur": row['valeur']}
    }
    geojson_data['features'].append(feature)

    # Ajouter un marqueur si la valeur de pollution est élevée
    if row['valeur'] > SEUIL_DE_VALEUR_ELEVEE:
        marker = Marker(location=(row['Y'], row['X']), draggable=False, title=f"Valeur: {row['valeur']}")
        high_value_markers.append(marker)

# Créer une couche GeoJSON pour les zones polluées
geojson_layer = GeoJSON(data=geojson_data, style={'color': 'red', 'opacity': 0.8, 'weight': 1.5})
carte.add_layer(geojson_layer)

# Ajouter les marqueurs à la carte
# Ajouter les marqueurs à la carte
for marker in high_value_markers:
    carte.add_layer(marker)


# Créer une légende
legend = widgets.VBox([
    widgets.HTML(value="<b>Légende</b>"),
   widgets.HTML(value=f'<div style="width: 0; height: 0; border-left: 10px solid transparent; border-right: 10px solid transparent; border-bottom: 20px solid red; display: inline-block;"></div> Valeur élevée (> {SEUIL_DE_VALEUR_ELEVEE})'),
    # Ajoutez d'autres lignes ci-dessus pour d'autres valeurs de pollution
])

# Afficher la carte avec la légende
display(widgets.HBox([carte, legend]))

# Trouver l'indice de la valeur maximale dans la colonne 'valeur'
indice_max = df['valeur'].idxmax()

# Obtenir les coordonnées X et Y pour l'emplacement de la valeur maximale
coordonnees_max = df.loc[indice_max, ['X', 'Y']]

print("Coordonnées de l'endroit avec la valeur la plus élevée:", coordonnees_max)



# %%
import requests
import matplotlib.pyplot as plt

# Remplacez le nom du jeu de données (dataset-name) et les filtres (facet et refine) par les valeurs appropriées
url_api = "https://public.opendatasoft.com/api/records/1.0/search/?dataset=donnees-synop-essentielles-omm&rows=100&facet=nom_reg&refine.nom_reg=Occitanie&facet=temps_present&refine.temps_present=Averse(s)%20de%20pluie"

# Faites la requête API
response = requests.get(url_api)
data = response.json()

# Exemple : Représentation d'une colonne "temps_present" en fonction de l'index
temps_present_values = [record['fields']['temps_present'] for record in data['records']]
index = range(len(temps_present_values))

# Créez un diagramme à barres
plt.bar(index, temps_present_values)
plt.xlabel('Index')
plt.ylabel('Temps Présent')
plt.title('Représentation du temps présent')
plt.show()

#%%
#données météo du site 

import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

# Remplacez le nom du jeu de données (dataset-name) et les filtres (facet et refine) par les valeurs appropriées
url_api = "https://public.opendatasoft.com/api/records/1.0/search/?dataset=donnees-synop-essentielles-omm&rows=100&facet=nom_reg&refine.nom_reg=Occitanie"

# Faites la requête API
response = requests.get(url_api)
data = response.json()

# Créer un DataFrame à partir des données
df = pd.DataFrame([record['fields'] for record in data['records']])

# Ajout d'une colonne temporelle pour la démonstration (à remplacer par votre propre colonne)
df['datetime'] = [datetime.now() - timedelta(days=random.randint(1, 365)) for _ in range(len(df))]

# Convertir la colonne 'datetime' en format datetime
df['datetime'] = pd.to_datetime(df['datetime'])

# Vérifier la structure des données
print(df.head())

# Représentation graphique
if 'temps_present' in df.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['temps_present'], marker='o', linestyle='-')
    plt.title('Représentation Temporelle du Dataset')
    plt.xlabel('Date et Heure')
    plt.ylabel('Temps Présent')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("La colonne 'temps_present' n'est pas présente dans le DataFrame.")

# %%
import pandas as pd
from ipyleaflet import Map, TileLayer, GeoJSON, Marker, AwesomeIcon
import ipywidgets as widgets
from IPython.display import display

# Remplacez 'votre_fichier.csv' par le chemin vers votre fichier CSV
chemin_fichier_csv = 'Mesure_mensuelle_Region_Occitanie_Polluants_Principaux.csv'
df = pd.read_csv(chemin_fichier_csv)

# Créer une colonne 'geometry' avec les coordonnées X et Y sous forme de GeoJSON
df['geometry'] = df.apply(lambda row: {"type": "Point", "coordinates": [row['X'], row['Y']]}, axis=1)

# Valeur seuil
SEUIL_DE_VALEUR_ELEVEE = 23  

# Créer une carte avec le service WMTS
carte = Map(center=(43.611015, 3.876733), zoom=9)

# Ajouter une couche WMTS à la carte
wmts_url = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/WMTS"
wmts_layer = TileLayer(url=wmts_url, name="WMTS Layer")
carte.add_layer(wmts_layer)

# Créer une GeoJSON FeatureCollection à partir des données de votre DataFrame
geojson_data = {
    "type": "FeatureCollection",
    "features": []
}

# Marqueurs pour les valeurs élevées
high_value_markers = []

for index, row in df.iterrows():
    feature = {
        "type": "Feature",
        "geometry": row['geometry'],
        "properties": {"nom_dept": row['nom_dept'], "valeur": row['valeur']}
    }
    geojson_data['features'].append(feature)

    # Ajouter un marqueur si la valeur de pollution est élevée
    if row['valeur'] > SEUIL_DE_VALEUR_ELEVEE:
        marker = Marker(location=(row['Y'], row['X']), draggable=False, title=f"Valeur: {row['valeur']}")
        high_value_markers.append(marker)

# Trouver l'indice de la valeur maximale dans la colonne 'valeur'
indice_max = df['valeur'].idxmax()

# Obtenir les coordonnées X et Y pour l'emplacement de la valeur maximale
coordonnees_max = df.loc[indice_max, ['X', 'Y']]

# Créer un marqueur pour l'emplacement avec la valeur la plus élevée
max_value_marker = Marker(location=(coordonnees_max['Y'], coordonnees_max['X']),
                         draggable=False,
                         title=f"Valeur maximale: {df.loc[indice_max, 'valeur']}",
                         icon=AwesomeIcon(name='star', marker_color='green', icon_color='white', spin=False))

# Ajouter les marqueurs à la carte
for marker in high_value_markers:
    carte.add_layer(marker)

# Créer une couche GeoJSON pour les zones polluées
geojson_layer = GeoJSON(data=geojson_data, style={'color': 'red', 'opacity': 0.8, 'weight': 1.5})
carte.add_layer(geojson_layer)

# Ajouter le marqueur pour la valeur maximale à la carte
carte.add_layer(max_value_marker)

# Créer une légende
legend = widgets.VBox([
    widgets.HTML(value="<b>Légende</b>"),
    widgets.HTML(value=f'<div style="width: 20px; height: 20px; background-color: red; display: inline-block; border-radius: 50%; margin-right: 5px;"></div> Valeur élevée (> {SEUIL_DE_VALEUR_ELEVEE})'),
    # Ajoutez d'autres lignes ci-dessus pour d'autres valeurs de pollution
])

# Afficher la carte avec la légende
display(widgets.HBox([carte, legend]))

#%%

#1-faire un dataframe des mesures annuelles et voir si il y a des améliorations du taux de pollution par ville au fil des années
#2- faire des graphiques originaux peut être 
#%%
import pandas as pd

# Remplacez 'votre_fichier.csv' par le chemin complet de votre fichier CSV
chemin_du_fichier = 'Mesure_annuelle_Region_Occitanie_Polluants_Principaux.csv'

# Charger le fichier CSV dans un DataFrame
dataframe = pd.read_csv(chemin_du_fichier)

# Afficher les premières lignes du DataFrame pour vérifier le chargement
print(dataframe.head())

#%%

#bis
#Je charge Mesure annuelle dans un df, et stock le nom des villes dans liste_villes

import pandas as pd

# Charger le fichier CSV dans un DataFrame
nom_fichier_csv = 'Mesure_annuelle_Region_Occitanie_Polluants_Principaux.csv'  # Remplacez par le chemin de votre fichier CSV
data = pd.read_csv(nom_fichier_csv)

# Assurez-vous que la colonne contenant les noms de ville est correcte
colonne_ville = 'nom_com'  # Remplacez par le nom de votre colonne

# Utiliser un ensemble pour stocker les noms de ville uniques
ensemble_villes = set(data[colonne_ville])

# Convertir l'ensemble en liste (si nécessaire)
liste_villes = list(ensemble_villes)

# Afficher la liste des villes uniques
print(liste_villes)

#32 villes 
len(liste_villes)

#Graphique pour Toulouse 

df['date_debut'] = pd.to_datetime(df['date_debut'])

# Tracer le graphique
plt.figure(figsize=(10, 6))
plt.plot(df['date_debut'], df['valeur'], marker='o')
plt.title('Taux de pollution à Toulouse au fil des années')
plt.xlabel('Année')
plt.ylabel('Taux de pollution (ug.m-3)')
plt.grid(True)

dossier_diapositives = "diapositives"
chemin_fichier_svg = os.path.join(dossier_diapositives, 'graph3.svg')
plt.savefig(chemin_fichier_svg, format='svg')

plt.show()

#Graphique pour Montpellier 



# %%
import pandas as pd
import matplotlib.pyplot as plt

# Remplacez 'votre_fichier.csv' par le chemin complet de votre fichier CSV
chemin_du_fichier = 'Mesure_annuelle_Region_Occitanie_Polluants_Principaux.csv'

# Charger le fichier CSV dans un DataFrame
dataframe = pd.read_csv(chemin_du_fichier)

# Filtrer les données pour Montpellier
montpellier_data = dataframe[dataframe['nom_com'] == 'Montpellier']

# Convertir la colonne 'Année' en type datetime pour s'assurer qu'elle est reconnue comme une année
montpellier_data['Année'] = pd.to_datetime(montpellier_data['Année'], format='%Y')

# Trier les données par année
montpellier_data = montpellier_data.sort_values(by='Année')

# Créer un graphique
plt.figure(figsize=(10, 6))
plt.plot(montpellier_data['Année'], montpellier_data['NO2'], label='NO2', marker='o')
plt.plot(montpellier_data['Année'], montpellier_data['PM10'], label='PM10', marker='o')
plt.plot(montpellier_data['Année'], montpellier_data['O3'], label='O3', marker='o')

# Ajouter des titres et des étiquettes
plt.title('Pollution à Montpellier au fil des années')
plt.xlabel('Année')
plt.ylabel('Concentration (µg/m³)')
plt.legend()

# Afficher le graphique
plt.show()

# %%

#Fonctionne pour mtp (polluant PM10)

import pandas as pd
import plotly.express as px

# Charger le fichier CSV dans un DataFrame
df = pd.read_csv('Mesure_annuelle_Region_Occitanie_Polluants_Principaux.csv')

# Filtrer les données pour la ville de Montpellier
montpellier_data = df[df['nom_com'] == 'MONTPELLIER']

# Convertir la colonne 'date_debut' en format datetime
montpellier_data['date_debut'] = pd.to_datetime(montpellier_data['date_debut'])

# Trier les données par date
montpellier_data = montpellier_data.sort_values(by='date_debut')

# Créer le graphique interactif avec Plotly Express
fig = px.line(montpellier_data, x='date_debut', y='valeur', markers=True,
              labels={'valeur': f'Concentration de {montpellier_data["nom_poll"].iloc[0]} (ug.m-3)'},
              title=f'Pollution à Montpellier au fil des années ({montpellier_data["nom_poll"].iloc[0]})')

# Ajouter des labels aux points de données
fig.update_traces(textposition='top center', text=montpellier_data['valeur'].round(2))

# Afficher le graphique interactif
fig.show()


# %%
#Commande utilse pour savoir les villes de mon fich csv

import pandas as pd

# Remplacez 'chemin/vers/votre/fichier.csv' par le chemin réel de votre fichier CSV
chemin_fichier_csv = ('Mesure_annuelle_Region_Occitanie_Polluants_Principaux.csv')

# Charger le DataFrame depuis le fichier CSV
df = pd.read_csv(chemin_fichier_csv)

# Utiliser drop_duplicates pour obtenir une liste unique de noms de ville
liste_villes_unique = df['nom_com'].drop_duplicates().tolist()

# Afficher la liste unique
print(liste_villes_unique)


#%%
#Graphique pour perpignan 

import pandas as pd
import plotly.express as px

# Charger le fichier CSV dans un DataFrame
df = pd.read_csv('Mesure_annuelle_Region_Occitanie_Polluants_Principaux.csv')

# Filtrer les données pour la ville de Perpignan
perpignan_data = df[df['nom_com'] == 'PERPIGNAN'].copy()  # Utiliser .copy() pour éviter le SettingWithCopyWarning

# Convertir la colonne 'date_debut' en format datetime
perpignan_data['date_debut'] = pd.to_datetime(perpignan_data['date_debut'])

# Trier les données par date
perpignan_data = perpignan_data.sort_values(by='date_debut')

# Créer un graphique en barres empilées avec Plotly Express
fig = px.bar(perpignan_data, x='date_debut', y='valeur', color='nom_poll',
             labels={'valeur': f'Concentration de pollution (ug.m-3)'},
             title='Répartition des concentrations de pollution à Perpignan par année',
             barmode='stack')

# Personnaliser le graphique
fig.update_layout(xaxis=dict(title='Année'), yaxis=dict(title='Concentration de pollution (ug.m-3)'))

# Afficher le graphique interactif
fig.show()

#%%
#Graphique pour Agde 

import pandas as pd
import plotly.express as px

# Charger le fichier CSV dans un DataFrame
df = pd.read_csv('Mesure_annuelle_Region_Occitanie_Polluants_Principaux.csv')

# Filtrer les données pour la ville d'Agde
agde_data = df[df['nom_com'] == 'AGDE'].copy()  # Utiliser .copy() pour éviter le SettingWithCopyWarning

# Convertir la colonne 'date_debut' en format datetime
agde_data['date_debut'] = pd.to_datetime(agde_data['date_debut'])

# Trier les données par date
agde_data = agde_data.sort_values(by='date_debut')

# Créer un graphique linéaire avec Plotly Express
fig = px.line(agde_data, x='date_debut', y='valeur', color='nom_poll',
              labels={'valeur': f'Concentration de pollution (ug.m-3)'},
              title='Évolution des concentrations de pollution à Agde par année')

# Personnaliser le graphique
fig.update_layout(xaxis=dict(title='Année'), yaxis=dict(title='Concentration de pollution (ug.m-3)'))

# Afficher le graphique interactif
fig.show()

#%%
# à rajouter pour Toulouse 
import pandas as pd
import plotly.express as px

# Charger le fichier CSV dans un DataFrame
df = pd.read_csv('Mesure_annuelle_Region_Occitanie_Polluants_Principaux.csv')

# Filtrer les données pour la ville de Toulouse
toulouse_data = df[df['nom_com'] == 'TOULOUSE'].copy()  # Utiliser .copy() pour éviter le SettingWithCopyWarning

# Convertir la colonne 'date_debut' en format datetime
toulouse_data['date_debut'] = pd.to_datetime(toulouse_data['date_debut'])

# Trier les données par date
toulouse_data = toulouse_data.sort_values(by='date_debut')

# Obtenir la valeur maximale de chaque polluant pour chaque année
max_values = toulouse_data.groupby(['date_debut', 'nom_poll'])['valeur'].max().reset_index()

# Créer un graphique en barres avec Plotly Express
fig = px.bar(max_values, x='date_debut', y='valeur', color='nom_poll',
             labels={'valeur': f'Concentration maximale de pollution (ug.m-3)'},
             title='Concentration maximale de pollution à Toulouse par année')

# Personnaliser le graphique
fig.update_layout(xaxis=dict(title='Année'), yaxis=dict(title='Concentration maximale de pollution (ug.m-3)'))

# Afficher le graphique interactif
fig.show()


# même si ça a atteint des seuils presque critiques la pollution totale de Toulouse diminue d'année en année, Bonne nouvelle ? 
# + interprétation des seuils critiques dictés par ministère écologie énérgie territoires 

#%%
#Villes à faire :


#Montauban
#Nimes 
#Lourdes 
#Lattes
#Beziers
#Rodez 
#Albi
#LUNEL-VIEL

#%%

