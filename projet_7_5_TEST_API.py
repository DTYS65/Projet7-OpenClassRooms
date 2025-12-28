# TESTS :
# Ouvrir le terminal sous anaconda, et changer de repertoire :
# cd C:\Users\jme1401\Desktop\Openclassrooms\7-Implémentez un modèle de scoring\Datas
# Taper : pytest projet_7_4_TEST_API.py : lance toutes les fonctions demarrant par test_


from fastapi import status
import requests
import json


API_URL = "http://127.0.0.1:8000/"


def test_welcome():
    """Teste la fonction welcome() de l'API."""
    response = requests.get(API_URL)
    assert response.status_code == status.HTTP_200_OK
    assert json.loads(response.content) == 'Welcome to the mega super API'


def test_check_client_id():
    """Teste la fonction check_client_id() de l'API avec un client faisant partie de la base de données X_test."""
    url = API_URL + str(192535)
    response = requests.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert json.loads(response.content) == True


def test_check_client_id_2():
    """Teste la fonction check_client_id() de l'API avec un client ne faisant pas partie de la base de données X_test."""
    url = API_URL + str(100000)
    response = requests.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert json.loads(response.content) == False


def test_get_prediction():
    """Teste la fonction get_prediction() de l'API."""
    url = API_URL + "prediction/" + str(192535)
    response = requests.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert json.loads(response.content) == 0.5451879132329674