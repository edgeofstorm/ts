# Generated by Django 3.1.3 on 2020-11-26 09:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('contribute', '0003_auto_20201126_1040'),
    ]

    operations = [
        migrations.RenameField(
            model_name='document',
            old_name='doc',
            new_name='document',
        ),
        migrations.AlterField(
            model_name='document',
            name='tokenized',
            field=models.TextField(blank=True, null=True, unique=True),
        ),
    ]
