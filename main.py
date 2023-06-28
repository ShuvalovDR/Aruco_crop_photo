import cv2  # import library for working with a picture
import numpy as np


def crop_image(img):
    dictionary = cv2.aruco.DICT_4X4_1000  # dictionary for our aruco markers
    aruco_dictionary = cv2.aruco.Dictionary_get(dictionary)  # dictionary
    aruco_parameters = cv2.aruco.DetectorParameters_create()  # searching parameters of markers standard
    (corners, ids, rejected) = cv2.aruco.detectMarkers(img, aruco_dictionary, parameters=aruco_parameters)  # get
    # corners coordinates of aruco markers in tuple
    x_min = np.array([])
    x_max = np.array([])
    y_min = np.array([])
    y_max = np.array([])
    for aruco_corners in corners:
        aruco_corners = np.hsplit(aruco_corners, 4)
        (x_top_left, y_top_left) = np.ravel(aruco_corners[0])
        (x_bottom_left, y_bottom_left) = np.ravel(aruco_corners[1])
        (x_bottom_right, y_bottom_right) = np.ravel(aruco_corners[2])
        (x_top_right, y_top_right) = np.ravel(aruco_corners[3])
        x_min = np.append(x_min, [x_top_left, x_bottom_left], axis=0)
        x_max = np.append(x_max, [x_bottom_right, x_top_right], axis=0)
        y_min = np.append(y_min, [y_bottom_left, y_bottom_right], axis=0)
        y_max = np.append(y_max, [y_top_right, y_top_left], axis=0)
    x_min = int(min(x_min))
    x_max = int(max(x_max))
    y_min = int(min(y_min))
    y_max = int(max(y_max))
    crp_image = img[y_min:y_max, x_min:x_max]
    return crp_image


image_1 = cv2.imread('first2d.jpg', cv2.IMREAD_COLOR)  # function imread is reading a picture
cropped_image_1 = crop_image(image_1)
image_2 = cv2.imread('second2d.jpg', cv2.IMREAD_COLOR)  # function imread is reading a picture
cropped_image_2 = crop_image(image_2)
image_3 = cv2.imread('third2d.jpg', cv2.IMREAD_COLOR)  # function imread is reading a picture
cropped_image_3 = crop_image(image_3)
cv2.imwrite('Cropped_image_1.jpg', cropped_image_1)
cv2.imwrite('Cropped_image_2.jpg', cropped_image_2)
cv2.imwrite('Cropped_image_3.jpg', cropped_image_3)
