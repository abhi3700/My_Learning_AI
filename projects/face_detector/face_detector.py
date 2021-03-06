import cv2
from random import randrange

# load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# img = cv2.imread('data/RDJ.jpg')		# read image
# img = cv2.imread('data/my_pic_2.jpg')		# read image
# img = cv2.imread('data/2_faces.png')		# read image
# img = cv2.imread('data/multiple_faces.jpg')     # read image
img = cv2.imread('data/female_washing_face.jpg')     # read image


# Convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect face
# captures the coordinates of a square formed on a face
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
# print(face_coordinates)

# Draw rectangles around the faces
# params: image, top left, bottom right, Green, thickness
# ----------------------------------------------------
# detect single face
# (x, y, w, h) = face_coordinates[0]
# cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
# ----------------------------------------------------
# detect multiple faces
'''
for i in range(len(face_coordinates)):
	x, y, w, h = face_coordinates[i];
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)   # Green color
	# randomly decide the color of rectangle for faces
	cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)
'''

# OR

for x, y, w, h in face_coordinates:
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)		# make the face green color
	# cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)          # randomly decide the color of rectangle for faces


# cv2.imshow("Face Detector", grayscaled_img)		# show image
cv2.imshow("Face Detector", img)		# show image
cv2.waitKey()		# for inifinite time, Press any key or close button to exit

