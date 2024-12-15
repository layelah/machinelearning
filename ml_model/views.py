from django.shortcuts import render
from django.http import JsonResponse
from .utils import my_model
import numpy as np


def predict(request):
    if my_model is None:
        return JsonResponse({'error': 'Le modèle n\'est pas disponible.'}, status=500)
    
    # Vérification des données reçues
    features = request.GET.getlist('features[]')
    if not features:
        return JsonResponse({'error': 'Veuillez fournir des données pour la prédiction.'}, status=400)
    
    try:
        # Convertir les données en tableau numpy
        features = np.array(features, dtype=float).reshape(1, -1)
        prediction = my_model.predict(features)
        return JsonResponse({'prediction': int(prediction[0])})
    except ValueError as e:
        return JsonResponse({'error': f'Données invalides : {str(e)}'}, status=400)




def predict_page(request):
    return render(request, 'ml_model/predict.html')
