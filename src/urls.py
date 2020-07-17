from django.urls import path, include
from .views import sentiment_extraction_view

urlpatterns = [
    path('', sentiment_extraction_view)
]