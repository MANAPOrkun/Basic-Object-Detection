import cv2
import matplotlib.pyplot as plt

car_img = cv2.imread('car_plate2.jpg')
cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')


def display(img):
    fixed = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(fixed, cmap='gray')
    plt.show()


def detect_and_blur_plate(img):
    plate_img = img.copy()
    place_rects = cascade.detectMultiScale(plate_img, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in place_rects:
        cropped = img.copy()[y:y + h, x:x + w]
        blurred = cv2.medianBlur(cropped, 15)
        plate_img[y:y + h, x:x + w] = blurred

    return plate_img


result = detect_and_blur_plate(car_img)
plt.imshow(result)
display(result)
