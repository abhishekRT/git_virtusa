# Generated by Django 3.1 on 2020-09-20 10:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0004_inpatient'),
    ]

    operations = [
        migrations.AlterField(
            model_name='inpatient',
            name='DiagnosisGroupCode',
            field=models.CharField(max_length=20, null=True),
        ),
    ]