from django.contrib.auth.decorators import login_required
from django.shortcuts import render

# Create your views here.
@login_required
def analytics(request):
    return render(request, 'pagina_de_control.html')