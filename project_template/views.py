from django.shortcuts import render
from django.shortcuts import render_to_response
from django.http import HttpResponse
from .models import Docs
from django.template import loader
from .form import QueryForm
from .test import find_similar
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

from django.contrib.auth.models import User
from django.http import JsonResponse
from django.contrib.auth.forms import UserCreationForm
from django.views.generic.edit import CreateView


# Create your views here.
def index(request):
    output_list = ''
    output=''
    data=""
    if request.GET.get('search'):
        search = request.GET.get('search')
        output_list = find_similar(search)
        paginator = Paginator(output_list, 10)
        page = request.GET.get('page')
        try:
            output = paginator.page(page)
        except PageNotAnInteger:
            output = paginator.page(1)
        except EmptyPage:
            output = paginator.page(paginator.num_pages)
        data = {
            'output': output_list
        }
    
    return render_to_response('project_template/index.html', 
                              {'output': output,
                               'magic_url': request.get_full_path(),
                               }
                               )


#return JsonResponse(data)
    #return HttpResponse("Hello, world. You're at the polls index.")



