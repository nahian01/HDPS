from django import forms
from django.contrib.auth import (
    authenticate,
    get_user_model,
    login,
    logout,
)
from .models import Disease
User = get_user_model()


class UserLoginForm(forms.Form):
    username = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput)

    def clean(self, *args, **kwargs):
        username = self.cleaned_data.get("username")
        password = self.cleaned_data.get("password")
        if username and password:
            user = authenticate(username=username, password=password)
            if not user:
                raise forms.ValidationError("username or password does not match")
            if not user.check_password(password):
                raise forms.ValidationError("Incorrect passsword")
            if not user.is_active:
                raise forms.ValidationError("This user is not longer active.")
        return super(UserLoginForm, self).clean(*args, **kwargs)


class DiseaseForm(forms.Form):
    age = forms.IntegerField(label="Age")
    sex = forms.BooleanField(label="Sex")
    cp = forms.IntegerField(label="Cp")
    trestbps = forms.IntegerField(label="Trestbps")
    chol = forms.IntegerField(label="Chol")
    fbs = forms.IntegerField(label="Fbs")
    restecg = forms.IntegerField(label="Restecg")
    thalach = forms.IntegerField(label="Thalach")
    exang = forms.IntegerField(label="Exang")
    oldpeak = forms.FloatField(label="Oldpeak")
    slope = forms.IntegerField(label="Slope")
    ca = forms.IntegerField(label="Ca")
    thal = forms.IntegerField(label="Thal")

    class Meta:
        model = Disease
        fields = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ]

    def clean(self, *args, **kwargs):
        return super(DiseaseForm, self).clean(*args, **kwargs)


class UserRegisterForm(forms.ModelForm):
    email = forms.EmailField(label='Email address')
    password = forms.CharField(widget=forms.PasswordInput)
    confirm_password = forms.CharField(widget=forms.PasswordInput, label='Confirm Password')

    class Meta:
        model = User
        fields = [
            'username',
            'email',
            'password',
            'confirm_password'
        ]

    def clean_confirm_password(self):
        email = self.cleaned_data.get('email')
        password = self.cleaned_data.get('password')
        confirm_password = self.cleaned_data.get('confirm_password')
        if password != confirm_password:
            raise forms.ValidationError("Password must match")
        email_qs = User.objects.filter(email=email)
        if email_qs.exists():
            raise forms.ValidationError(
                "This email has already been registered"
            )
        return email
