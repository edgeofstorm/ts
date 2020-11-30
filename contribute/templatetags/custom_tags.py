from django import template 
register = template.Library() 

from ..models import Document

@register.simple_tag
def get_token_by_id(index): 
    return Document.objects.get(id= index).token