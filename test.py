import pytest
from app import app, index
import json


def test_index():
    assert (index() == '''<h1>Ceci est mon API FLASK conçu pour interroger à distance la base de données HomeCredit</h1>
            <p>Attention l'API est à l'état prototype. Ne le présentez pas au client avant l'aval de votre direction</p>''')


test_data = [
    ("126208", 0),
    ("333039", 0),
    ("208", -1)
]

@pytest.mark.parametrize("SK_ID_CURR, expected", test_data)
def test_get_score(SK_ID_CURR, expected):
    response = app.test_client().get(f'/viability/?id={SK_ID_CURR}')
    res = json.loads(response.data.decode('utf-8')).get("score")
    assert (res == expected)



expected_features_list =[
    ("5", ["EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_1", "BURO_CREDIT_ACTIVE_Active_MEAN", "BURO_DAYS_CREDIT_UPDATE_MEAN"]),
    ("3", ["EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_1"])
]

@pytest.mark.parametrize("n,expected", expected_features_list)
def test_get_features(n, expected):
    response = app.test_client().get(f'/best_features/?n={n}')
    resp = json.loads(response.data.decode('utf-8')).get("n_features")
    assert (resp == expected)


# pour exécuter le code : pytest test.py