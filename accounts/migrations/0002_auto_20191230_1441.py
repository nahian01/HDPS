# Generated by Django 3.0.1 on 2019-12-30 14:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='disease',
            name='bp',
        ),
        migrations.AddField(
            model_name='disease',
            name='bp_high',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='disease',
            name='bp_low',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='disease',
            name='bmi',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='disease',
            name='bs',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='disease',
            name='hr',
            field=models.IntegerField(),
        ),
    ]