import cv2
import numpy as np
from pyzbar.pyzbar import decode

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set frame width
cap.set(4, 480)  # Set frame height

# Load authorized barcodes
with open('myDataFile.text') as f:
    myDataList = f.read().splitlines()

while True:
    # Capture frame-by-frame
    success, img = cap.read()
    img = cv2.imread("auth.jpg")
    if not success:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to highlight barcodes
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Decode barcodes from the thresholded image
    barcodes = decode(thresh)
    for barcode in barcodes:
        myData = barcode.data.decode('utf-8')
        print(myData)

        # Check if barcode data is authorized
        if myData in myDataList:
            myOutput = 'Authorized'
            myColor = (0, 255, 0)  # Green for authorized
        else:
            myOutput = 'Un-Authorized'
            myColor = (0, 0, 255)  # Red for unauthorized

        # Draw bounding box around detected barcode
        pts = np.array(barcode.polygon, np.int32)
        if len(pts) == 4:  # Ensure the barcode has four corners
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, myColor, 2)

        # Draw text indicating the authorization status
        pts2 = barcode.rect
        cv2.putText(img, myOutput, (pts2[0], pts2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, myColor, 2)

    # Display the resulting frame
    cv2.imshow('Result', img)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
