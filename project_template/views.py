from django.shortcuts import render
from django.shortcuts import render_to_response
from django.http import HttpResponse
from .models import Docs
from django.template import loader
from .form import QueryForm
from .test import original_query
from .test import requery
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.contrib.auth.forms import UserCreationForm
from django.views.generic.edit import CreateView


def index(request):
    output = []
    search = request.GET.get('search','')
    request_type = (request.META["HTTP_ACCEPT"].split(",")[0])
    if (request_type == "application/json"):
        output = original_query(search)
        return JsonResponse(output, content_type="application/json", safe=False)
        
    elif (request_type == "text/html"):
        return render_to_response('project_template/index.html', 
                              {
                               'search_params': search
                              })

