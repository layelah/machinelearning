import joblib
import os

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'decision_tree_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier modèle est introuvable : {model_path}")
    return joblib.load(model_path)

# Chargement sécurisé du modèle
try:
    my_model = load_model()
except FileNotFoundError as e:
    print(e)
    my_model = None
