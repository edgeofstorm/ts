from django.urls import path
from . import views

app_name='contribute'

urlpatterns = [
    # path("", views.HomeView.as_view(), name="home"),
    path("", views.SummaryFormView.as_view(), name="contribute")
]
