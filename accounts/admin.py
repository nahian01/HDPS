from django.contrib import admin
from .models import (
    Disease,
    KNN,
    DTree,
    RForest,
    Adaboost,
    LRegression,
    SVM,
    Nbayes,
    XGBoost,
)
# Register your models here.

admin.site.register(Disease)
admin.site.register(KNN)
admin.site.register(DTree)
admin.site.register(RForest)
admin.site.register(Adaboost)
admin.site.register(LRegression)
admin.site.register(SVM)
admin.site.register(Nbayes)
admin.site.register(XGBoost)
