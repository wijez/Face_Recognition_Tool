import numpy as np
import face_recognition
import cv2

# Load the image files using OpenCV
imgBich_path = "training/lizkimcuong.jpg"
imgLiz_path = "training/bichphuong2.jpg"

try:
    imgBich = cv2.imread(imgBich_path)
    imgLiz = cv2.imread(imgLiz_path)
    print("Images loaded successfully using OpenCV.")
except Exception as e:
    print(f"Error loading images with OpenCV: {e}")
    exit()

# Convert images to RGB format
imgBich_rgb = cv2.cvtColor(imgBich, cv2.COLOR_BGR2RGB)
imgLiz_rgb = cv2.cvtColor(imgLiz, cv2.COLOR_BGR2RGB)

# Locate faces in the RGB images
try:
    faceLocBich = face_recognition.face_locations(imgBich_rgb)
    if len(faceLocBich) > 0:
        print(f"Bich face located: {faceLocBich}")
        # Encode faces
        encodeBich = face_recognition.face_encodings(imgBich_rgb, faceLocBich)[0]
        for (top, right, bottom, left) in faceLocBich:
            cv2.rectangle(imgBich_rgb, (left, top), (right, bottom), (255, 0, 255), 2)
    else:
        print("No face found in Bich's image.")
except Exception as e:
    print(f"Error locating Bich's face: {e}")

try:
    faceLocLiz = face_recognition.face_locations(imgLiz_rgb)
    if len(faceLocLiz) > 0:
        print(f"Liz face located: {faceLocLiz}")
        # Encode faces
        encodeLiz = face_recognition.face_encodings(imgLiz_rgb, faceLocLiz)[0]
        for (top, right, bottom, left) in faceLocLiz:
            cv2.rectangle(imgLiz_rgb, (left, top), (right, bottom), (255, 0, 255), 2)
    else:
        print("No face found in Liz's image.")
except Exception as e:
    print(f"Error locating Liz's face: {e}")

# Compare encodings
results = face_recognition.compare_faces([encodeBich], encodeLiz)
faceDis = face_recognition.face_distance([encodeBich], encodeLiz)
print(results, faceDis)
cv2.putText(imgLiz_rgb, f"{results} {round(faceDis[0], 2)}", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
# Display the images with detected faces
cv2.imshow("Bich Image", imgBich_rgb)
cv2.imshow("Liz Image", imgLiz_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
