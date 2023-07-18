from django.urls import path,include
from .views import captionView


urlpatterns = [
    path('' , captionView , name='caption')
]