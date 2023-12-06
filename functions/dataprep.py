import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt


def get_data(time):
    if time == "tj":
        url = "https://services9.arcgis.com/7Sr9Ek9c1QTKmbwr/arcgis/rest/services/Mesure_horaire_(30j)_Region_Occitanie_Polluants_Reglementaires_1/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=json"
    elif time == "m":
        url = "https://services9.arcgis.com/7Sr9Ek9c1QTKmbwr/arcgis/rest/services/mesures_occitanie_mensuelle_poll_princ/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=json"
    response = requests.get(url)
    return response.json()


def extract_data_to_df(data: dict) -> pd.DataFrame:
    """extracts important data from dict and returns dataframe with data"""

    # get important data
    cities = []
    starts = []
    ends = []
    val = []
    poll_ue = []
    typo = []
    infl = []
    nom_poll = []

    for element in data["features"]:
        cities += [element.get("attributes").get("nom_com")]
        starts += [element.get("attributes").get("date_debut")]
        ends += [element.get("attributes").get("date_fin")]
        val += [element.get("attributes").get("valeur")]
        poll_ue += [element.get("attributes").get("id_poll_ue")]
        typo += [element.get("attributes").get("typologie")]
        infl += [element.get("attributes").get("influence")]
        nom_poll += [element.get("attributes").get("nom_poll")]

    # construct df from data
    df_data = pd.DataFrame()
    df_data["city"] = cities
    df_data["start"] = starts
    df_data["end"] = ends
    df_data["values"] = val
    df_data["poll_ue"] = poll_ue
    df_data["nom_poll"] = nom_poll
    df_data["influence"] = infl
    df_data["typology"] = typo

    # convert timestamps to dates
    df_data["start"] = df_data["start"].apply(lambda x: dt.fromtimestamp(x / 1e3))
    df_data["end"] = df_data["end"].apply(lambda x: dt.fromtimestamp(x / 1e3))

    return df_data


def get_unique_cities(df) -> list:
    """takes df and returns unique list of cities"""
    return df.city.unique()


def data_prep(time):
    data = get_data(time=time)
    df = extract_data_to_df(data=data)
    return df

