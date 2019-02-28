from hyperlpr import *
import cv2


image = cv2.imread("G:/demo4.png")
print(HyperLPR_PlateRecogntion(image))
cv2.imshow("", image)
cv2.waitKey(0)
