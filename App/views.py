from django.shortcuts import render, HttpResponse
from . import read_data


# Create your views here.

def index(request):
    context = {
        'data' : read_data.df

    }
    return HttpResponse( render(request,'index.html', context) )