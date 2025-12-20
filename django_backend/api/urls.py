from django.urls import path
from . import views

urlpatterns = [
    path('health/', views.health_check, name='health_check'),
    path('predict/', views.predict, name='predict'),
    path('logs/', views.get_logs, name='get_logs'),
]
