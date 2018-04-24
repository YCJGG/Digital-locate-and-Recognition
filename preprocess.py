import os 
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
# resize and save 
# downsample 4x

# images = glob.glob(r'.\picture\*.jpg')
# k = 0
# for image in images:
   
#     im = cv2.imread(image)
#     print(im.shape)
#     im = cv2.resize(im,(864,1152))
#     # cv2.imshow('r',im)
#     # cv2.waitKey(0)
#     cv2.imwrite(str(k)+'.png',im)
#     k+=1

def resize(rawimg):  # resize img to 28*28
    fx = 28.0 / rawimg.shape[0]
    fy = 28.0 / rawimg.shape[1]
    fx = fy = min(fx, fy)
    img = cv2.resize(rawimg, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    outimg = np.ones((28, 28), dtype=np.uint8) * 255
    w = img.shape[1]
    h = img.shape[0]
    x = int((28 - w) // 2)
    y = int((28 - h) // 2)
    
    outimg[y:y+h, x:x+w] = img
    return outimg


images = glob.glob(r'./*.png')

#image = images[1]
image = './0.png'
img=cv2.imread(image)  
GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
bw = cv2.adaptiveThreshold(GrayImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 20)
img2, ctrs, hier = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
cv2.imshow('r',bw)
cv2.waitKey(0)
for rect in rects:
    x, y, w, h = rect
    hw = float(h) / w
    if (w < 100) & (h < 100) & (h > 10) & (w > 10) & (0.1 < hw) & (hw < 6):
        cv2.rectangle(GrayImage, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # print(x,y,w,h)
    # roi = GrayImage[y:y+h, x:x+w]
    # hw = float(h) / w
    # if (w < 200) & (h < 200) & (h > 10) & (w > 10) & (0.5 < hw) & (hw < 5):
    #     res = resize(roi)
cv2.imshow('r',GrayImage)
cv2.waitKey(0)