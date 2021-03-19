from django.shortcuts import render
from django.http import HttpResponse
from .models import Document


def DocumentShows(request , ID):
    document = Document.objects.get(pk = ID)
    return HttpResponse(f"Document Name:  {document.name}")
# Create your views here.
