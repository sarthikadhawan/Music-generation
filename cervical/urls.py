"""cervical URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from usurvey import views

urlpatterns = [
    url(r'^home/',views.home, name="home"),
        url(r'^home1/',views.home1, name="home1"),

    url(r'^play/',views.play, name="play"),
    url(r'^trial/', views.trial,  name="trial"),
    url(r'^graphs/', views.graphs,  name="graphs"),
    url(r'^about/', views.about,  name="about"),

    url(r'^LSTM/', views.LSTM,  name="LSTM"),
    url(r'^RBM/', views.RBM,  name="RBM"),
    url(r'^GRU/', views.GRU,  name="GRU"),


    url(r'^admin/', admin.site.urls),

    #url(r'^usurvey/home', views.home, name="home"),
   

]
