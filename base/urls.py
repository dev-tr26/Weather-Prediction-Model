from django.urls import path
from . import views

urlpatterns = [
    path('',views.predict_weather,name='predict_weather'),
]