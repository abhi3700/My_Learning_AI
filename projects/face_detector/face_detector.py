import cv2

# load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# img = cv2.imread('data/RDJ.jpg')		# read image
# img = cv2.imread('data/my_pic.jpg')		# read image
img = cv2.imread('data/my_pic_2.jpg')		# read image

# Convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect face
# captures the coordinates of a square formed on a face
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
# print(face_coordinates)

# Draw rectangles around the faces
# params: image, top left, bottom right, Green, thickness
(x, y, w, h) = face_coordinates[0]
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# cv2.imshow("Face Detector", grayscaled_img)		# show image
cv2.imshow("Face Detector", img)		# show image
cv2.waitKey()		# for inifinite time, Press any key or close button to exit

