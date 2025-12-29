# LANCEMENT :

# Imports
from fastapi import FastAPI
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import uvicorn
import shap
import joblib


def fn_load_joblib(filename, filepath):
    return joblib.load(f'{filepath}{filename}') 


# Create a FastAPI instance
app = FastAPI(
    title="Projet n°7",
    version="1.0.0",
    description="Implémentez un modèle de scoring."
)


# Répertoires
path_dir = "./"

X_train = pd.read_csv(path_dir + 'df_datas_train_reduit_cleaned.csv')
X_test = pd.read_csv(path_dir + 'df_datas_test_reduit_cleaned.csv')

cols = X_train.select_dtypes(['float64']).columns
X_train_scaled = X_train.copy()
X_train_scaled[cols] = StandardScaler().fit_transform(X_train[cols])

cols = X_test.select_dtypes(['float64']).columns
X_test_scaled = X_test.copy()
X_test_scaled[cols] = StandardScaler().fit_transform(X_test[cols])

# Loading the model and tree_explainer
# model_final = fn_load_joblib("model_final.joblib", path_dir)
# tree_explainer = shap.TreeExplainer(model_final['classifier'], approximate=True)


# Functions
@app.get('/')
def welcome():
    """
    Welcome message.
    :param: None
    :return: Message (string).
    """
    return 'Bienvenue dans AAA le projet n°7 d\'Openclassrooms : Implémentez un modèle de scoring'


@app.get("/test_user/{user_id}")
def test_user(user_id: int):
    return {"user_id": user_id}


@app.get('/{client_id}')
def check_client_id(client_id: int) -> bool:
    """
    Customer search in the database
    :param: client_id (int)
    :return: message (string).
    """
    if client_id in list(X_test['SK_ID_CURR']):
        return True
    else:
        return False


@app.get('/client/{client_id}')
def get_client(client_id: int):
    """
    Give data client.
    :param: client_id (int)
    :return: datas of the client.
    """
    client_data = X_test[X_test['SK_ID_CURR'] == client_id]
    return client_data.to_json()


@app.get('/prediction/{client_id}')
def get_prediction(client_id: int):
    """
    Calculates the probability of default for a client.
    :param: client_id (int)
    :return: probability of default (float).
    """
    model_final = fn_load_joblib("model_final.joblib", path_dir)

    client_data = X_test[X_test['SK_ID_CURR'] == client_id]
    client_data = client_data.drop('SK_ID_CURR', axis=1)
    prediction = model_final.predict_proba(client_data)[0][1]
    return prediction


@app.get('/clients_similaires/{client_id}')
def get_data_voisins(client_id: int):
    """ Calcul les plus proches voisins du client_id et retourne le dataframe de ces derniers.
    :param: client_id (int)
    :return: dataframe de clients similaires (json).
    """
    features = list(X_test_scaled.columns)
    features.remove('SK_ID_CURR')

    # Création d'une instance de NearestNeighbors
    nn = NearestNeighbors(n_neighbors=10, metric='euclidean')

    # Entraînement du modèle sur les données
    nn.fit(X_train_scaled[features])
    reference_id = client_id
    reference_observation = X_test_scaled[X_test_scaled['SK_ID_CURR'] == reference_id][features].values
    indices = nn.kneighbors(reference_observation, return_distance=False)
    df_voisins = X_train.iloc[indices[0], :]

    return df_voisins.to_json()


@app.get('/shaplocal/{client_id}')
def shap_values_local(client_id: int) -> dict:
    """ Calcul les shap values pour un client.
        :param: client_id (int)
        :return: shap values du client (json).
        """
    model_final = fn_load_joblib("model_final.joblib", path_dir)
    tree_explainer = shap.TreeExplainer(model_final['classifier'], approximate=True)

    client_data = X_test_scaled[X_test_scaled['SK_ID_CURR'] == client_id]
    client_data = client_data.drop('SK_ID_CURR', axis=1)
    shap_val = tree_explainer(client_data)[0]
    return {
        'shap_values': shap_val.values.tolist(),
        'base_value': shap_val.base_values,
        'data': client_data.values.tolist(),
        'feature_names': client_data.columns.tolist()
    }


@app.get('/shap/')
def shap_values() -> dict:
    """ Calcul les shap values de l'ensemble du jeu de données
    :param:
    :return: shap values
    """
    model_final = fn_load_joblib("model_final.joblib", path_dir)
    tree_explainer = shap.TreeExplainer(model_final['classifier'], approximate=True)

    shap_val = tree_explainer.shap_values(X_test_scaled.drop('SK_ID_CURR', axis=1))
    return {
        'shap_values_0': shap_val[0].tolist(),
        'shap_values_1': shap_val[1].tolist()
    }


# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)
