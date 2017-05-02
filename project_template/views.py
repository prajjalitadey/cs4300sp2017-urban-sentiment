from django.shortcuts import render
from django.shortcuts import render_to_response
from django.http import HttpResponse
from .models import Docs
from django.template import loader
from .form import QueryForm
from .test import original_query
from .test import requery
from .test import handle_click
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.contrib.auth.forms import UserCreationForm
from django.views.generic.edit import CreateView
import ast


def index(request):
    output = []
    search = request.GET.get('search','')
    extra = request.GET.get('extra','')
    request_type = (request.META["HTTP_ACCEPT"].split(",")[0])

    if (request_type == "application/json" and request.method == "GET"):

      try:
        d = extra
        if 'relevant' in d:
          requery = requery(d)
          return JsonResponse(requery, content_type="application/json", safe=False)
        elif '//name//' in d:
          clicked = handle_click(d)
          return JsonResponse(clicked, content_type="application/json", safe=False)
        else:
          print("not there")
      except:
        print("ajax error")

      output = original_query(search)
      return JsonResponse(output, content_type="application/json", safe=False)
        
    elif (request_type == "text/html"):
        return render_to_response('project_template/index.html', 
                              {
                               'search_params': search
                              })

