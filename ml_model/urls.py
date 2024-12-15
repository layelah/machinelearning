from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_page, name='predict_page'),
    path('predict/', views.predict, name='predict'),
]
