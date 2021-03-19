from django.urls import path

from . import views

urlpatterns = [
    path('Document/<int:ID>', views.DocumentShows, name='DocumentShows'),
]