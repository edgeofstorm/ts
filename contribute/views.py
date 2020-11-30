from django.shortcuts import render
from .forms import SummaryForm
from .models import Summary, Document
from . import tokenize
from django.views.generic.edit import FormView
from django.contrib.auth.models import User
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.exceptions import ValidationError


class SummaryFormView(LoginRequiredMixin, FormView):
    template_name = 'contribute.html'
    form_class = SummaryForm
    success_url = '/contribute/'

    def form_valid(self, form):
        candidate = form.save(commit=False)
        print(self.request.POST)
        candidate.user = User.objects.get(username=self.request.user.username)  # use your own profile here
        print(candidate.user)
        doc = Document.objects.get(id=self.request.POST['document'])
        selected_sentences = self.request.POST.getlist('selected')
        binary = []
        summary = []
        tokenized = tokenize.run(doc.document)
        for i, sentence in enumerate(tokenized):
            if str(i) in selected_sentences:
                binary.append(1)
                summary.append(sentence)
            else:
                binary.append(0)
        candidate.summary = summary
        candidate.binary_representation = binary
        candidate.save()
        return super().form_valid(form)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        docs = Document.objects.all()
        context["docs"] = Document.objects.all()
        for i, doc in enumerate(docs,start=1):
            context[f"doc{i}"] = doc.token
        # print(context)
        return context
    
    # validate unique together