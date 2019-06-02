import numpy as np
import cv2
import os
import imutils
import random

root_dir = os.path.dirname(__file__)
image_directory = root_dir+ '/image/'
train_data = []

#get training data
for filename in sorted(os.listdir(image_directory)):
    if filename.endswith(".png"):
        img = cv2.imread(image_directory + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = imutils.resize(img, height=50,width=50)
        train_data.append(img) 


train_data = np.asarray(train_data, dtype=np.float32)
print(train_data.shape)
train_data = np.reshape(train_data,(20,2500))
print(train_data.shape)

#mark training label
train_label = np.array([0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9], dtype=int)


#create SVM model
model = cv2.ml.SVM_create()
model.setType(cv2.ml.SVM_C_SVC)
model.setKernel(cv2.ml.SVM_LINEAR)
model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))
model.train(train_data, cv2.ml.ROW_SAMPLE, train_label)

#save model
model.save("model.xml")


#predict image
test_data = []
filename = random.choice([x for x in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, x))])
img = cv2.imread(image_directory + '/' + filename)
img = imutils.resize(img, height=50,width=50)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

test_data.append(img) 
test_data = np.asarray(test_data, dtype=np.float32)
test_data = np.reshape(test_data,(1,2500))

#load model
model2 = cv2.ml.SVM_load("model.xml")
_, y_val = model2.predict(test_data)
print("prediction result")
print(y_val)

cv2.imshow("Image",img)
cv2.waitKey(0)  
cv2.destroyAllWindows()
