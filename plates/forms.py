from django import forms

class UploadForm(forms.Form):
    image = forms.ImageField(required=False)
    image_url = forms.URLField(required=False)




