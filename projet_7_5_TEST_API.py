# TESTS :
# Ouvrir le terminal sous anaconda, et changer de repertoire :
# cd C:\Users\jme1401\Desktop\Openclassrooms\7-Implémentez un modèle de scoring\Datas
# Taper : pytest projet_7_5_TEST_API.py : lance toutes les fonctions demarrant par test_


from fastapi import status
import requests
import json


# API_URL = "http://127.0.0.1:8000/"
API_URL = "https://projet-7-modele-de-scoring-6b3669013dac.herokuapp.com/"


def test_welcome():
    """Teste la fonction welcome() de l'API."""
    response = requests.get(API_URL)
    assert response.status_code == status.HTTP_200_OK
    assert json.loads(response.content) == 'Bienvenue sur le projet Openclassrooms n°7 : Implémentez un modèle de scoring'


def test_check_client_id():
    """Teste la fonction check_client_id() de l'API avec un client faisant partie de la base de données X_test."""
    url = API_URL + str(168959)
    response = requests.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert json.loads(response.content) == True


def test_check_client_id_2():
    """Teste la fonction check_client_id() de l'API avec un client ne faisant pas partie de la base de données X_test."""
    url = API_URL + str(100000)
    response = requests.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert json.loads(response.content) == False


def test_get_client():
    """Teste la fonction get_client() de l'API."""
    url = API_URL + "client/" + str(168959)
    response = requests.get(url)
    assert response.status_code == status.HTTP_200_OK


def test_get_prediction():
    """Teste la fonction get_prediction() de l'API."""
    url = API_URL + "prediction/" + str(168959)
    response = requests.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert json.loads(response.content) == 0.0847574849837666


def test_get_prediction():
    """Teste la fonction get_prediction() de l'API."""
    url = API_URL + "prediction/" + str(425400)
    response = requests.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert json.loads(response.content) == 0.24716962315451516


def test_get_data_voisins():
    """Teste la fonction get_data_voisins() de l'API."""
    url = API_URL + "clients_similaires/" + str(425400)
    response = requests.get(url)
    assert response.status_code == status.HTTP_200_OK