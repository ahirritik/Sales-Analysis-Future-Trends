from django.urls import path
from . import views

urlpatterns = [
    path('', views.ml_analysis, name='ml_analysis'),
    path('tableau/', views.tableau_dashboard, name='tableau_dashboard'),
]
