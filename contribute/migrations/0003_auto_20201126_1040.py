# Generated by Django 3.1.3 on 2020-11-26 07:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('contribute', '0002_auto_20201126_1036'),
    ]

    operations = [
        migrations.AlterField(
            model_name='summary',
            name='binary_representation',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
