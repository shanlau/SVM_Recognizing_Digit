from imutils import contours
import imutils
import cv2
import numpy as np
 

image = cv2.imread("sample.png")
 
image = imutils.resize(image, height=500) 
contrast = cv2.convertScaleAbs(image, alpha=5, beta=90)
gray = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (15, 15))
img = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

cnts = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
#cv2.drawContours(image,cnts,-1,(0, 255, 0), 2)
boundingRect = []
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    if (h > image.shape[0]*0.3 and h <image.shape[0]*0.9):    
        cv2.rectangle(image,(x,y),(x+w,y+h),(0, 0, 255), 2)
        boundingRect.append([x,y,w,h])

#cv2.drawContours(image,cnts,-1,(0, 255, 0), 2)
boundingRect = sorted(boundingRect, key=lambda boundingRect: boundingRect[0])

standard = 50
test_data = []
for r in boundingRect:
    (x, y, w, h) = r
    d = imutils.resize(threshold[y:y + h, x:x + w], height=standard,width=standard) 
    print(d.shape)
    tmp_h, tmp_w = d.shape
    #padding white pixel to make the digit in square scale
    ptop = int((standard * 2 - tmp_h) / 2)
    pbuttom = standard * 2  - ptop - tmp_h
    pleft = int((standard * 2  - tmp_w) / 2)
    pright = standard * 2 - pleft - tmp_w
    print(str(ptop) + '-' + str(pbuttom) + '-' + str(pleft) + '-' + str(pright))    
    if (ptop > 0 and pbuttom > 0 and pleft > 0 and pright > 0):
        constant = cv2.copyMakeBorder(d,ptop,pbuttom,pleft,pright,cv2.BORDER_CONSTANT,value=[255,255,255])
        print(constant.shape)    
        reshape = imutils.resize(constant, height=standard,width=standard)     
        #cv2.imshow("Image", reshape)   
        #cv2.waitKey(0)      
        test_data.append(reshape)    
    

test_data = np.asarray(test_data, dtype=np.float32)
print(test_data.shape)
test_data = np.reshape(test_data,(test_data.shape[0],standard*standard))
print(test_data.shape)

#load model
model2 = cv2.ml.SVM_load("model.xml")
_, y_val = model2.predict(test_data)
print("prediction result")
print(np.int_(y_val))


#cv2.imshow("Image", image)   
#cv2.waitKey(0)  
#cv2.destroyAllWindows()