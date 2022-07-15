# import the necessary modules
from matplotlib.cbook import flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

#INITIALIZING LEARNING RATE,EPOCHS,BATCH SIZE
INIT_LR = 1e-4     #0.0001
EPOCHS = 20
BS = 32

#GIVING UP THE DIRECTORY FOR THE DATASET
DIRECTORY = r"D:\REAL_TIME_FACE_MASK_DETECTOR\dataset"
CATEGORIES = ["with_mask","without_mask"]

#TAKING THE LIST OF IMAGES PRESENT IN THE DATASET
print("[MESSAGE]-> LOADING IMAGES.....")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY,category)
    for img in os.listdir(path):
        img_path = os.path.join(path,img)
        image = load_img(img_path,target_size=(224,224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

#perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data , dtype = "float32")
labels = np.array(labels)

(trainX , testX , trainY, testY) = train_test_split(data,labels,test_size = 0.30,stratify = labels
                                                        , random_state = 46)

#CONSTRUCTING TRAINING IMAGE GENERATOR FOR DATA AUGMENTATION(TECHNIQUE TO ARTIFICIALLY CREATE NEW TRAINING DATA FROM EXISTING TRAINING DATA)
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

#LOAD THE MOBILENETV2 NETWORK , ENSURUNG THE HEAD FC LAYER SETS ARE LEFT OFF
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

#CONSTRUCT THE HEAD OF THE MODEL THAT WILL BE PLACED ON THE TOP OF THE BASE MODEL
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128,activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2,activation="softmax")(headModel)

#PLACE THE HEAD FC MODEL ON TOP OF THE BASE MODEL(THIS WILL BECOME THE 
# ACTUAL MODE WE WILL TRAIN)
model = Model(inputs = baseModel.input , outputs = headModel)

#LOOP OVER ALL LAYERS IN THE BASE MODEL AND FREEZE THEM SO THEY WILL
#NOT BE UPDATED DURING THE FIRST TRAINING PROCESS
for layer in baseModel.layers:
    layer.trainable = False

#COMPILE THE MODEL
print("[MESSAGE]->COMPILING MODEL...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss = "binary_crossentropy",optimizer = opt ,
                        metrics = ["accuracy"])

#TRAIN THE HEAD OF THE NETWORK
print("[MESSAGE]->TRAINING HEAD...")
H = model.fit(aug.flow(trainX,trainY,batch_size=BS),
            steps_per_epoch=len(trainX)//BS,
            validation_data=(testX,testY),
            validation_steps = len(testX)//BS,
            epochs = EPOCHS)

#MAKE PREDICTION ON THE TESTING SET
print("[MESSAGE]->EVALUATING NETWORK...")
predIdxs = model.predict(testX,batch_size = BS)

#FOR EACH IMAGE IN THE TESTING SET WE NEED TO FIND THE INDEX OF THE
#LABEL WITH CORRESPONDING LARGEST PREDICTED PROBABLITITY
predIdxs = np.argmax(predIdxs, axis = 1)

#SHOW A NICELY FORMATTED CLASSIFICATION REPORT
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

#SERIALIZE THE MODEL TO DISK
print("[MESSAGE]->SAVING MASK DETECTOR MODEL...")
model.save("mask_detector.model",save_format="h5")

#PLOT THE TRAINING LOSS AND ACCURACY
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N),H.history["loss"],label = "train_loss")
plt.plot(np.arange(0,N),H.history["val_loss"],label="val_loss")
plt.plot(np.arange(0,N),H.history["accuracy"],label = "train_acc")
plt.plot(np.arange(0,N),H.history["val_accuracy"],label="val_acc")
plt.title("Training Loss And Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc = "lower left")
plt.savefig("plot.png")