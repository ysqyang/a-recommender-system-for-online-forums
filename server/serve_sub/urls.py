from django.urls import path

from . import views

urlpatterns = [
    path('', views.serve_recoms_for_subject, 
         name='serve_recoms_for_subject'),
]