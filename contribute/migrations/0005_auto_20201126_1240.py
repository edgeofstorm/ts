# Generated by Django 3.1.3 on 2020-11-26 09:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('contribute', '0004_auto_20201126_1235'),
    ]

    operations = [
        migrations.AlterField(
            model_name='document',
            name='tokenized',
            field=models.TextField(blank=True, null=True),
        ),
    ]