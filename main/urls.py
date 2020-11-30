from django.urls import path, include
from . import views
from django.views.generic import TemplateView

urlpatterns = [
    path("", views.a, name="main"),
    path('summary/', TemplateView.as_view(template_name="summary.html")),
    path('parameters/', views.parameters, name="parameters"),
    path("contribute/", include('contribute.urls')),
    path("register/", views.register, name="register"),
]