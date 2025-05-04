import cv2
import numpy as np
import pytesseract
import urllib.request
from django.shortcuts import render
from .forms import UploadForm
from django.core.files.storage import FileSystemStorage
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def process_image(image):
    def resize_image(image, width):
        h, w = image.shape[:2]
        ratio = width / w
        dim = (width, int(h * ratio))
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    image = resize_image(image, 500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(filtered, 30, 200)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    license_plate = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            license_plate = approx
            break

    if license_plate is not None:
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [license_plate], 0, 255, -1)
        new_img = cv2.bitwise_and(image, image, mask=mask)
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        cropped = gray[topx:bottomx+1, topy:bottomy+1]
    else:
        cropped = gray

     # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Tesseract config with whitelist
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(thresh, config=config)

    #DEBUG: print raw extracted text
    print("Extracted text:", repr(text))
    return text.strip()

def upload_image(request):
    extracted_text = None
    image_url = None

    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Handle uploaded file
            if form.cleaned_data['image']:
                image = form.cleaned_data['image']
                fs = FileSystemStorage()
                filename = fs.save(image.name, image)
                file_url = fs.url(filename)
                image_path = fs.path(filename)
                img = cv2.imread(image_path)
                extracted_text = process_image(img)
                image_url = file_url

            # Handle image from URL
            elif form.cleaned_data['image_url']:
                url = form.cleaned_data['image_url']
                resp = urllib.request.urlopen(url)
                img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                extracted_text = process_image(img)
                image_url = url

    else:
        form = UploadForm()

    return render(request, 'upload.html', {
        'form': form,
        'extracted_text': extracted_text,
        'image_url': image_url,
    })
