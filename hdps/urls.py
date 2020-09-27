from django.contrib import admin
from django.urls import path, include
from accounts.views import home_page
urlpatterns = [
    path("", home_page, name="home"),
    path('admin/', admin.site.urls),
    path('users/', include(("accounts.urls", "accounts"), namespace='accounts')),
]
