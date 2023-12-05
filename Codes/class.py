import pandas as pd

class CSVAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.load_data()

    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Data loaded successfully from {self.file_path}")
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
        except Exception as e:
            print(f"Error loading data: {e}")

    def show_head(self, n=5):
        if self.data is not None:
            return self.data.head(n)
        else:
            return "No data loaded. Use load_data() method first."

    def describe_data(self):
        if self.data is not None:
            return self.data.describe()
        else:
            return "No data loaded. Use load_data() method first."

    def filter_data(self, column, value):
        if self.data is not None:
            try:
                filtered_data = self.data[self.data[column] == value]
                return filtered_data
            except KeyError:
                return f"Column '{column}' not found in the dataset."
        else:
            return "No data loaded. Use load_data() method first."

# Exemple d'utilisation de la classe
if __name__ == "__main__":
    # Créer une instance de la classe avec le chemin du fichier CSV
    csv_analyzer = CSVAnalyzer("votre_fichier.csv")

    # Charger les données
    csv_analyzer.load_data()

    # Afficher les premières lignes du DataFrame
    print(csv_analyzer.show_head())

    # Afficher les statistiques descriptives
    print(csv_analyzer.describe_data())

    # Filtrer les données
    filtered_data = csv_analyzer.filter_data('nom_com', 'TOULOUSE')
    print(filtered_data)
