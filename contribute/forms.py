from django.forms import ModelForm
from .models import Summary, Document
from django.contrib.auth.models import User

# Create the form class.
class SummaryForm(ModelForm):
    class Meta:
        model = Summary
        fields = ['document']