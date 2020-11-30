from django import forms

class InputForm(forms.Form):
    your_name = forms.CharField(label='Your name', max_length=255)

# class InputForm(forms.Form):
#     title = forms.CharField(label='Title', max_length=255)
#     input = forms.Textarea(label='Input')
    