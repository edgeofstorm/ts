from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.contrib import messages
from . import tokenize
# Create your models here.

class Document(models.Model):

    title = models.CharField(max_length=255)
    document = models.TextField(unique=False)
    tokenized = models.TextField(null=True, unique=False, blank=True)
    token = models.JSONField(default=list, null=True, blank=True, unique=False)
    sentence_count = models.IntegerField(default=0)

    def __str__(self):
        return f'{self.id} - {self.title}'
    


class Summary(models.Model):

    user = models.ForeignKey(User,on_delete=models.CASCADE)
    document = models.ForeignKey(Document,on_delete=models.CASCADE)
    summary  = models.TextField(null=False, blank=False)
    binary_representation = models.CharField(null=True, blank=True, max_length=255)
    #binary_representation = models.JSONField(default=list)

    class Meta:
        unique_together = (('user','document'),)
    
    def __str__(self):
        return f'{self.user} - {self.document.id}'    


# signal for creating binary rep
def summary_post_save_receiver(sender, instance, *args, **kwargs):
    if instance.summary:
        instance.binary_representation = ''
        for sentence in instance.document.tokenized:
            if sentence in instance.summary:
                instance.binary_representation += '1 '
                # instance.binary_representation.append(1)
            else:
                instance.binary_representation += '0 '
        
                # instance.binary_representation.append(0)

post_save.connect(receiver=summary_post_save_receiver,sender=Summary)

# signal for creating tokenized document
# call .py for creating tokenized output from original document
def document_post_save_receiver(sender, instance, *args, **kwargs):
    if instance.document:
        # instance.tokenized = tokenize.run(instance.document)
        # doc, created = Document.objects.get_or_create(document= instance.document)
        tokenized = tokenize.run(instance.document)
        strsent=''
        for sentence in tokenized:
            # for i in range(10):
            #     sentence.replace(f'[{i}]','')
            strsent+=sentence + "~"    
        sentence_count = len(tokenized)
        Document.objects.filter(document=instance.document).update(token=tokenized, tokenized = strsent, sentence_count=sentence_count)
        
        #instance.save()
        
post_save.connect(receiver=document_post_save_receiver,sender=Document)


