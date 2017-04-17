from django.conf.urls import url

from . import views

#app_name = 'pt'
urlpatterns = [
    url(r'^$', views.index, name='index')
    #url(r'^signup/$', views.SignUpView.as_view(), name='signup')
]

