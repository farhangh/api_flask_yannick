# c'est notre api_flask
import flask
from flask import request, jsonify
# from flask import Flask, request, jsonify
from scoring.utilities import read_data, load_model, get_features_importance, display_features_importance

import json

import base64
from io import BytesIO

from lime.lime_tabular import LimeTabularExplainer


#data = read_data(path="path="data/df_train.csv")
data = read_data()
print("data_train : ", data.head(), "data.shape : ", data.shape)

features = data.drop(["SK_ID_CURR", "TARGET"], axis=1).columns.tolist()

X = data[features].values
print("X : ", X, "data.shape : ", len(X))

model = load_model(path="data/ml_lr_f10.pkl")
model = load_model()
print("model :\n", model)


# instantiate Flask object
app = flask.Flask(__name__)
app.config["Debug"] = True

# test local : http://127.0.0.1:5000/
@app.route("/", methods=['GET'])
def index():
    return '''<h1>Ceci est mon API FLASK conçu pour interroger à distance la base de données HomeCredit</h1>
            <p>Attention l'API est à l'état prototype. Ne le présentez pas au client avant l'aval de votre direction</p>'''


# http://127.0.0.1:5000/list_IDs/
@app.route('/list_IDs/')
def identity():
    ids = data['SK_ID_CURR']
    lst_ids = ids.tolist()
    ids = {"identiy": lst_ids}

    return jsonify(json.loads(json.dumps(ids)))


# http://127.0.0.1:5000/best_features/?n=10
@app.route("/best_features/")
def get_features():
    n = request.args.get('n', type=int)
    features = data.drop(["SK_ID_CURR", "TARGET"], axis=1).columns.tolist()
    df_features_importance = get_features_importance(model,features).head(n)
    list_features = df_features_importance['name'].tolist()
    dct_features = {"n_features": list_features}
    return jsonify(json.loads(json.dumps(dct_features)))


# http://127.0.0.1:5000/features_importance_global/?n=10
@app.route("/features_importance_global/")
def display_features():
    n = request.args.get('n', type=int)
    df_features_importance = get_features_importance(model, features).head(n)
    fig=display_features_importance(df_features_importance)
    # Réglez la taille de la figure ici
    fig.set_size_inches(27, 10)
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    image_encoding = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{image_encoding}'/>"



# http://127.0.0.1:5000/viability/?id=126208
# http://127.0.0.1:5000/viability/?id=333039
@app.route('/viability/')
def get_score():
    id = float(request.args.get('id'))
    id_data = data[data['SK_ID_CURR']==id]
    idx = id_data.index
    if(id_data.shape[0]==0):
        score = {"score":-1}
    else:
        predict = model.predict_proba(X[idx])[:,1][0]
        predict = 1 if predict > 0.4 else 0
        score = {"score":predict}

    return jsonify(json.loads(json.dumps(score)))

# http://127.0.0.1:5000/lime/?n=10&id=126208
# http://127.0.0.1:5000/lime/?n=15&id=333039
@app.route("/lime/")
def display_lime(data=data):
    n = request.args.get('n', type=int)
    id = float(request.args.get('id'))
    id_data = data[data['SK_ID_CURR'] == id]
    idx = id_data.index

    if (id_data.shape[0] == 0):
        return "ID invalide."

    else:
        explainer = LimeTabularExplainer(X, mode="classification",
                                            class_names=['Viable', 'Risque'],
                                            feature_names=features,
                                            discretize_continuous = False
                                        )

        explanation = explainer.explain_instance(X[idx].reshape(len(features),), model.predict_proba, num_features=n)

        return explanation.as_html()

        # Sauvegarder et afficher
        '''with plt.style.context("ggplot"):
            fig = explanation.as_pyplot_figure()
            # Save it to a temporary buffer.
            buf = BytesIO()
            fig.savefig(buf, format="png")
            # Embed the result in the html output.
            image_lime = base64.b64encode(buf.getbuffer()).decode("ascii")

        return f"<img src='data:image/png;base64,{image_lime}'/>" 
        '''


#app.run()
app.run(host='0.0.0.0', port=8080)



