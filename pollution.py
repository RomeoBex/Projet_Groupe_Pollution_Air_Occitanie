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

# Utilisez la méthode value_counts() pour compter les occurrences de chaque valeur dans la colonne 'valeur'
valeurs_counts = dataframe['valeur'].value_counts()

# Créez un diagramme à barres
valeurs_counts.plot(kind='bar', color='skyblue')

# Ajoutez des étiquettes et un titre
plt.xlabel('Valeur')
plt.ylabel('Nombre d\'occurrences')
plt.title('Occurrences des valeurs dans la colonne "valeur"')

# Affichez le diagramme
plt.show()

# %%
