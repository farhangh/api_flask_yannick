import pickle
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def read_data(path="data/df_train.csv"):
    return pd.read_csv(path).reset_index(drop=True)

def load_model(path="data/ml_lr_f10.pkl"):
    with open(path, "rb") as model_file:
        return pickle.load(model_file)  # se référer au notebook 3-Classification_pretsbancaire_top20_20230803 section enregistrement du meilleur modèle

def get_features_importance(model, features):
    features_importance = model.best_estimator_['estimator'].coef_[0]
    print("features_importance: ", features_importance)
    df_features_importance = pd.DataFrame(data=zip(features , features_importance),columns=['name','coef'])
    df_features_importance = df_features_importance.sort_values(by='coef', ascending=False, key=abs).reset_index(drop=True)

    return df_features_importance

def display_features_importance(df):
    # Définir le style de seaborn (facultatif)
    sns.set(style='whitegrid')

    # Afficher le DataFrame avec seaborn
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x='coef', y='name', orient='h')
    plt.xlabel('Coef')
    plt.ylabel('Categories')
    plt.title('Features-importance')

    return fig
