from django.shortcuts import render, redirect
from .forms import InputForm
from main.summarizer import run
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse, reverse_lazy
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java, isJVMStarted
#from jpype import shutdownJVM
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.contrib.staticfiles.storage import staticfiles_storage


ZEMBEREK_PATH = staticfiles_storage.path('zemberek-full.jar')
    
if not isJVMStarted():
    startJVM(getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH), convertStrings=True)

TurkishMorphology = JClass('zemberek.morphology.TurkishMorphology')
morphology = TurkishMorphology.createWithDefaults()
PerceptronNer = JClass('zemberek.ner.PerceptronNer')
NamedEntity = JClass('zemberek.ner.NamedEntity')
ner_model_path: java.nio.file.Paths = (
    java.nio.file.Paths.get(staticfiles_storage.path('my-model'))
)
ner_model: PerceptronNer = (
    PerceptronNer.loadModel(ner_model_path, morphology)
)


def main(request,summary):
    return render(request, 'main.html', {'form':InputForm, 'summary':summary})


def a(request):

    summary=''
    if request.method == 'POST':
        # print(request.POST)
        title = request.POST.get('title')
        input = request.POST.get('input')
        compression = float(request.POST.get('compression'))
        comp = 1 - compression
        if comp == 0:
            comp=0.1
        
        summary, info = run(title=title,input=input,compression_ratio=comp, morphology=morphology, ner_model=ner_model)
        return render(request, 'main.html', {'form':InputForm, 'summary':summary, 'input':input, 'title':title, 'compression':int(compression*100), 'info':info})
        # shutdownJVM()
        # if form.is_valid(): 
    return render(request, 'main.html', {'form':InputForm, 'summary':summary})


def parameters(request):
    summary=''
    info=''
    if request.method == 'POST':
        # print(request.POST)
        title = request.POST.get('title')
        input = request.POST.get('input')
        compression = float(request.POST.get('compression'))
        coefs={'ts':float(request.POST.get('title-similarity')),
            'sp':float(request.POST.get('sentence-position')),
            'tfidf':float(request.POST.get('tfidf')),
            'sl':float(request.POST.get('sentence-length')),
            'ner':float(request.POST.get('ner'))}
        population=int(request.POST.get('population-size'))
        iter_count=int(request.POST.get('iteration-count'))
        mutation_rate = float(request.POST.get('mutation-rate'))
        crossover_rate = float(request.POST.get('crossover-rate'))
        selection = request.POST.get('selection-method')
        elitist = True
        if request.POST.get('elitist') == 'False':
            elitist=False
        comp = 1 - compression
        if comp == 0:
            comp=0.1
        
        summary, info = run(title=title,
                input=input,
                compression_ratio=comp,
                morphology=morphology,
                ner_model=ner_model,
                coeffs=coefs,
                elitist=elitist,
                population=population,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                selection=selection)

        return render(request, 'parameters.html', {'form':InputForm, 'summary':summary, 'input':input, 'title':title, 'compression':int(compression*100), 'infos':info})

    return render(request, 'parameters.html', {'form':InputForm, 'summary':summary, 'infos':info})


def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}!')
            return redirect('contribute:contribute')
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form':form})