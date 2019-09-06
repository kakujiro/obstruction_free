import cv2, matplotlib
import numpy as np
import matplotlib.pyplot as plt

# read an image
img = cv2.imread('rose.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#print(img)
plt.imshow(img)
#plt.show()

# convert image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#print(gray_img)
plt.imshow(gray_img)
#plt.show()

# find average per row, assuming image is already in the RGB format
average_color_per_row = np.average(img, axis = 0)

#find average across average per row
average_color = np.average(average_color_per_row, axis = 0)

# convert back to uint8
average_color = np.uint8(average_color)
#print(average_color)

# create 100 * 100 pixel image with average color value
average_color_img = np.array([[average_color] * 100] * 100, np.uint8)
plt.imshow(average_color_img)
#plt.show()


################################################################################


# threshold for image, with threshold 60
_, threshold_img = cv2.threshold(gray_img, 60, 255, cv2.THRESH_BINARY)
threshold_img = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2RGB)
plt.imshow(threshold_img)
#plt.show()

# threshold for hue channel in blue range
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

blue_min = np.array([100, 100, 100], np.uint8)
blue_max = np.array([140, 255, 255], np.uint8)
threshold_blue_img = cv2.inRange(img_hsv, blue_min, blue_max)

threshold_blue_img = cv2.cvtColor(threshold_blue_img, cv2.COLOR_GRAY2RGB)
plt.imshow(threshold_blue_img)
#plt.show()

# masking
upstate = cv2.imread('nature.jpg')
upstate_hsv = cv2.cvtColor(upstate, cv2.COLOR_BGR2HSV)
upstate_rgb = cv2.cvtColor(upstate_hsv, cv2.COLOR_HSV2RGB)
plt.imshow(upstate_rgb)
#plt.show()

#get mask of pixels that are in blue range
mask_inverse = cv2.inRange(upstate_hsv, blue_min, blue_max)
plt.imshow(mask_inverse)
#plt.show()

# inverse mask to get parts that are not blue
mask = cv2.bitwise_not(mask_inverse)
plt.imshow(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
#plt.show()

# convert single channel mask back into 3 channels
mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

#perform bitwise and on mask to obtain cut-out image that is not blue
masked_upstate = cv2.bitwise_and(upstate, mask_rgb)
#plt.show()

# replace the cut-out with white
masked_replace_white = cv2.addWeighted(masked_upstate, 1, \
cv2.cvtColor(mask_inverse, cv2.COLOR_GRAY2RGB), 1, 0)

plt.imshow(cv2.cvtColor(masked_replace_white, cv2.COLOR_RGB2BGR))
#plt.show()


################################################################################


img = cv2.imread('town.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
#gaussian blurring with a 5*5 kernel
img_blur_small = cv2.GaussianBlur(img, (15,15), 0)
plt.imshow(img_blur_small)
plt.show()
