import cv2

# load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('data/RDJ.jpg')		# read image

# Convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Face Detector", grayscaled_img)		# show image
cv2.waitKey()		# for inifinite time, Press any key or close button to exit

