# c'est notre api_flask
import numpy as np
import flask
from flask import request, jsonify
# from flask import Flask, request, jsonify
from scoring.utilities import read_data, load_model, get_features_importance, display_features_importance

import json

import base64
from io import BytesIO

from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#data = read_data(path="path="data/df_train.csv")
data = read_data()
print("data_train : ", data.head(), "data.shape : ", data.shape)

features = data.drop(["SK_ID_CURR", "TARGET"], axis=1).columns.tolist()
print("liste des features:", features, "size_features:", len(features))

# X = data[features]
# print("X.head() : ", X.head(), "type X :", type(X), "X.shape :", X.shape)

X = data.drop('TARGET', axis=1)
print("X : ", X.head(), "X.shape : ", X.shape)

X_126208 = X.iloc[3].values
print("X[126208.0] :", X_126208)

model = load_model(path="data/ml_lr_f10.pkl")
model = load_model()
print("model :\n", model)


# instantiate Flask object
app = flask.Flask(__name__)
app.config["Debug"] = True

@app.route("/", methods=['GET'])
def index():
    return '''<h1>Ceci est mon API FLASK conçu pour interroger à distance la base de données HomeCredit</h1>
            <p>Attention l'API est à l'état prototype. Ne le présentez pas au client avant l'aval de votre direction</p>'''


# test local : http://127.0.0.1:5000/viability/?id=126208
@app.route('/viability/')
def get_score():
    id = int(request.args.get('id'))
    id_data = data[data['SK_ID_CURR']==id]
    idx = id_data.index
    if(id_data.shape[0]==0):
        score = {"score":-1}
    else:
        predict = model.predict_proba(X[idx])[:,1][0]
        predict = 1 if predict > 0.4 else 0
        score = {"score":predict}

    return jsonify(json.loads(json.dumps(score)))

# test local : http://127.0.0.1:5000/list_IDs/
@app.route('/list_IDs/')
def identity():
    ids = data['SK_ID_CURR']
    lst_ids = ids.tolist()
    ids = {"identiy": lst_ids}

    return jsonify(json.loads(json.dumps(ids)))


# http://127.0.0.1:5000/best_features/?n=5
@app.route("/best_features/")
def get_features():
    n = request.args.get('n', type=int)
    features = data.drop(["SK_ID_CURR", "TARGET"], axis=1).columns.tolist()
    df_features_importance = get_features_importance(model,features).head(n)
    list_features = df_features_importance['name'].tolist()
    dct_features = {"n_features": list_features}
    return jsonify(json.loads(json.dumps(dct_features)))

# http://127.0.0.1:5000/features/?n=5
@app.route("/features/")
def display_features():
    n = request.args.get('n', type=int)
    df_features_importance = get_features_importance(model, features).head(n)
    fig=display_features_importance(df_features_importance)

    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    image_encoding = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{image_encoding}'/>"



# http://127.0.0.1:5000/lime/?n=5&id=126208
# @app.route("/lime/")
def display_lime(data=data):
    # n = request.args.get('n', type=int)
    n=20

    #id = float(request.args.get('id'))
    #data=data.reset_index(drop=True)
    #id_data = data[data['SK_ID_CURR'] == id]
    #idx = int(id_data.index[0])

    #if (id_data.shape[0] == 0):
    #    return "ID invalide."

    #else:
        # Obtenir les explications de LIME
        #lst_features = get_features_importance(model, features).head(n)["name"].tolist()
        #print("liste des features importance :", lst_features)
    features = data.drop(["SK_ID_CURR", "TARGET"], axis=1).columns.tolist()
    print("liste des features:", features, "size_features:", len(features))
    X_2 = X.drop('SK_ID_CURR', axis=1)
    # scaler = StandardScaler()
    # X_2_scaled = scaler.fit_transform(X_2)
    X_2_126208=X_2.iloc[3].values
    print( "X_2_126208 ", X_2_126208, "shape", X_2_126208.shape )
    print("type  X_2_126208 ", type(X_2_126208))
    print( " ******  X_2_126208.reshape ", X_2_126208.reshape(len(features),).shape )
    print( "N: ", n)
    explainer = LimeTabularExplainer(X_2.values, mode="classification",
                                            class_names=['Viable', 'Risque'],
                                            feature_names=features
                                        )
        # explanation = explainer.explain_instance(X[idx].values.reshape(len(features),) , model.predict_proba, num_features=n)
    explanation = explainer.explain_instance(X_2_126208, model.predict_proba, num_features=n)
    # print('****** explanation ****** :', explanation)



       # Sauvegarder et servir l'image
    '''with plt.style.context("ggplot"):
        fig = explanation.as_pyplot_figure()
            # Save it to a temporary buffer.
        buf = BytesIO()
        fig.savefig(buf, format="png")
            # Embed the result in the html output.
        image_lime = base64.b64encode(buf.getbuffer()).decode("ascii")

    return f"<img src='data:image/png;base64,{image_lime}'/>"
    '''

# app.run()

display_lime()



