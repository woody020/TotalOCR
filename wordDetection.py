import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
img = cv2.imread('input\ouy3d.jpeg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#print(pytesseract.image_to_string(img))

### Detecting Characters
#hImg,wImg,  = img.shape
print(img.shape)
boxes=pytesseract.image_to_data(img)
print(boxes)

cv2.imshow('Result',img)
cv2.waitKey(0)
