# Generated by Django 3.1.3 on 2020-11-26 10:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('contribute', '0007_document_token'),
    ]

    operations = [
        migrations.AlterField(
            model_name='document',
            name='token',
            field=models.JSONField(default=list, null=True),
        ),
    ]