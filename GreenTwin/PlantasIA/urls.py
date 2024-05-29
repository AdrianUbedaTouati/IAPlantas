from django.contrib import admin
from django.urls import path, include
from PlantasIA import views

urlpatterns = [
    path('analytics', views.analytics)
]
