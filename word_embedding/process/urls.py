from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.index, name='predict_context'),
    url(r'^train_brown_corpus/', views.train_using_nltk_brown_corpus, name='train_brown_corpus'),
]