from django.urls import path

from . import views

urlpatterns = [
    path('', views.serve_recommendations, name='serve_recommendations'),
]