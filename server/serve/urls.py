from django.urls import path

from . import views

urlpatterns = [
    path('<topic_id>', views.serve, name='serve'),
]