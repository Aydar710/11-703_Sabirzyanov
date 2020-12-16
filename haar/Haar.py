import cv2
import matplotlib.pyplot as plt

face_utils_font = cv2.FONT_HERSHEY_SIMPLEX

casc_path = "opencv/data/haarcascades/haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_faces(img):
    return face_cascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE
    )


def show_img(img):
    plt.figure(figsize=(16, 12))
    plt.imshow(img, cmap='gray')
    plt.show()


def draw_rectangles_around_faces(faces, img):
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 3)


def detect_faces_on_img(img_path):
    gray_img = cv2.imread(img_path, 0)

    show_img(gray_img)

    faces = detect_faces(gray_img)

    draw_rectangles_around_faces(faces, gray_img)

    show_img(gray_img)


if __name__ == '__main__':
    img_path_random_faces = 'faces.jpg'
    img_path_uncle_bob = 'bob.jpg'

    detect_faces_on_img(img_path_random_faces)
    detect_faces_on_img(img_path_uncle_bob)
