from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='predict_context'),
    path('train_brown_corpus/', views.train_using_nltk_brown_corpus, name='train_brown_corpus'),
]