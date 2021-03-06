# Generated by Django 3.0.1 on 2020-02-17 12:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0007_auto_20200217_1241'),
    ]

    operations = [
        migrations.AlterField(
            model_name='adaboost',
            name='accuracy',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='adaboost',
            name='con_00',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='adaboost',
            name='con_01',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='adaboost',
            name='con_10',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='adaboost',
            name='con_11',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='disease',
            name='age',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='disease',
            name='ca',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='disease',
            name='chol',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='disease',
            name='cp',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='disease',
            name='exang',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='disease',
            name='fbs',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='disease',
            name='restecg',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='disease',
            name='sex',
            field=models.BooleanField(),
        ),
        migrations.AlterField(
            model_name='disease',
            name='slope',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='disease',
            name='thal',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='disease',
            name='thalach',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='disease',
            name='trestbps',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='dtree',
            name='accuracy',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='dtree',
            name='con_00',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='dtree',
            name='con_01',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='dtree',
            name='con_10',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='dtree',
            name='con_11',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='knn',
            name='accuracy',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='knn',
            name='con_00',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='knn',
            name='con_01',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='knn',
            name='con_10',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='knn',
            name='con_11',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='lregression',
            name='accuracy',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='lregression',
            name='con_00',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='lregression',
            name='con_01',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='lregression',
            name='con_10',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='lregression',
            name='con_11',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='nbayes',
            name='accuracy',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='nbayes',
            name='con_00',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='nbayes',
            name='con_01',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='nbayes',
            name='con_10',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='nbayes',
            name='con_11',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='rforest',
            name='accuracy',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='rforest',
            name='con_00',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='rforest',
            name='con_01',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='rforest',
            name='con_10',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='rforest',
            name='con_11',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='svm',
            name='accuracy',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='svm',
            name='con_00',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='svm',
            name='con_01',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='svm',
            name='con_10',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='svm',
            name='con_11',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='xgboost',
            name='accuracy',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='xgboost',
            name='con_00',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='xgboost',
            name='con_01',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='xgboost',
            name='con_10',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='xgboost',
            name='con_11',
            field=models.IntegerField(),
        ),
    ]
