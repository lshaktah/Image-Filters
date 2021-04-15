import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy import ndimage as nd
from skimage.filters import sobel
from skimage.filters import sobel_h

# Open the image_____
file = "13APR21 Gel"
# file = "cell"
img = cv2.imread(file + ".JPG")
img = cv2.resize(img, (int(img.shape[0] / 8), int(img.shape[1] / 5)))  # This lets me resize it

# This is a Black and White filter
imgBW = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sobel_h filter
Sobel_h = sobel_h(imgBW)

# This applies a Guassian Filter (Blur) with sigma being how blurred_____
gausian_filtered_image = nd.gaussian_filter(imgBW, sigma=5)

# This adds an "edges" filter_____
sobel_filtered_image = sobel(imgBW)

# This adds an entropy detection filter, its needs disk to work, which is why it was imported in the beginning_____
entropy_filtered_image = entropy(imgBW, disk(1))

# This writes the filtered images to HD
cv2.imwrite("0_B&W_" + file + ".jpg", imgBW)
cv2.imwrite("0_Gausian_" + file + ".jpg", gausian_filtered_image)
cv2.imwrite("0_Sobel_" + file + ".jpg", sobel_filtered_image)
cv2.imwrite("0_Entropy_" + file + ".jpg", entropy_filtered_image)
cv2.imwrite("00Sobel_h_image.jpg", Sobel_h)

# This shows some of the images that save "black" but do have some content
cv2.imshow("Sobel_h", Sobel_h)
cv2.imshow("Entropy", entropy_filtered_image)
cv2.imshow("Sobel", sobel_filtered_image)

cv2.waitKey()  # This keeps the windows open when you use img.show()
cv2.destroyWindow()  # This closes them but not really
