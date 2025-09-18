from django.urls import path
from . import views
from django.shortcuts import render

urlpatterns = [
    path("", views.student_view, name="student_home"),
    path("teacher/", views.teacher_view, name="teacher_home"),
]