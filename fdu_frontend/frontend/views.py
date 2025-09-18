from django.shortcuts import render


def student_view(request):
    return render(request, "frontend/index.html")

def teacher_view(request):
    return render(request, "frontend/teacher.html")