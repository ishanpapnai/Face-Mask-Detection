# IMPORTING THE NECESSARY MODULES
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_predict_mask(frame, faceNet, maskNet):
	# TAKING THE DIMENSIONS OF THE BLOCK
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# INITIALIZE FACE,LOCATIONS AND PREDICTIONS OF MASK OR WITHOUT MASK
	faces = []
	locs = []
	preds = []

	# LOOP OVER THE DETECTIONS
	for i in range(0, detections.shape[2]):
		#EXTRACTING THE PROBABILITY THROUGH DETECTIONS
		confidence = detections[0, 0, i, 2]

		# FILTERING OUT WEAK DETECTIONS 
		if confidence > 0.5:
			# COMPUTE THE (x,y) COORDINATES OF THE BOUNDING BOX FOR THE OBJECT
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ENSURING THE BOUNDING BLOCK FALLS WITHIN THE FRAME
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# EXTRACT THE FACE ROI,CONVERT IT TO RGB FROM BGR CHANNEL
            #RESIZE IT TO 224*224 AND PREPROCESS IT
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# ADD FACE AND BOX TO THERE REPECTIVE LIST
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# IF ONE FACE IS DETECTED
	if len(faces) > 0:
		# FOR FASTER WE MAKE BATCH PREDICTIONS ON *ALL* FACES RATHER THAN ONE BY ONE
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# RETURN A 2-TUPLE OF THE FACE LOCATIONS AND THERE CORRESPONDING LOCATIONS
	return (locs, preds)

# LOADING THE FACE DETECTOR MODEL FROM DISK
prototxtPath = r"face_detection\deploy.prototxt"
weightsPath = r"face_detection\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# LOADING THE MASK DETECTION MODEL(i.e.,Main.py)FROM THE DISK
maskNet = load_model("mask_detector.model")

#INTIALIZE THE VIDEO STREAM
print("[MESSAGE] starting...")
vs = VideoStream(src=0).start()

# LOOPING OVER ALL THE FRAMES FROM STREAM
while True:
	#GRAB THE FRAME FROM THE STREAM AND RESIZE IT TO MAX 300 PIXELS
	frame = vs.read()
	frame = imutils.resize(frame, width=300)

	# DETECT FACE WHETHER WEARING MASK OR NOT
	(locs, preds) = detect_predict_mask(frame, faceNet, maskNet)

	# LOOP OVER THE DETECTED FACE AND THERE LOCATIONS
	for (box, pred) in zip(locs, preds):
		
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# DETERMINE THE CLASS LABEL AND COLOR AND THE MESSAGE TO POP UP
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# THE PROBABILITY IN THE LABEL
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# DISPLAY THE LABEL AND BOUNDING BOX RECTANGLE ON THE OUTPUT
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# THE OUTPUT FRAME
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# IF THE "1" IS PRESSED BREAK THE LOOP
	if key == ord("c"):
		break

# CLEAN THE WINDOW
cv2.destroyAllWindows()
vs.stop()