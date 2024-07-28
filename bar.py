import cv2
import numpy as np
from pyzbar.pyzbar import decode

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    bilateral_filtered = cv2.bilateralFilter(blurred, 9, 75, 75)
    return bilateral_filtered

def detect_barcodes(img):
    return decode(img)

def annotate_image(img, barcodes, myDataList):
    for barcode in barcodes:
        myData = barcode.data.decode('utf-8')
        myOutput = 'Authorized' if myData in myDataList else 'Un-Authorized'
        myColor = (0, 255, 0) if myData in myDataList else (0, 0, 255)
        
        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, myColor, 5)
        pts2 = barcode.rect
        cv2.putText(img, myOutput, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, myColor, 2)
    return img

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

with open('myDataFile.text') as f:
    myDataList = f.read().splitlines()

while True:
    success, img = cap.read()
    img= cv2.imread("bar.jpg")
    if not success:
        break

    # Preprocessing
    processed_img = preprocess_image(img)

    # Barcode Detection
    barcodes = detect_barcodes(processed_img)

    # Annotate Image
    annotated_img = annotate_image(img, barcodes, myDataList)

    # Display Result
    cv2.imshow('Result', annotated_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
