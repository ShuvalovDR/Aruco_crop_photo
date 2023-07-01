import cv2  # import library for working with a picture
import numpy as np


def fix_aruco(img, aruco_dict):
    aruco_dictionary = cv2.aruco.Dictionary_get(aruco_dict)  # dictionary
    aruco_parameters = cv2.aruco.DetectorParameters_create()  # searching parameters of markers standard
    (corners, ids, rejected) = cv2.aruco.detectMarkers(img, aruco_dictionary, parameters=aruco_parameters)  # get
    # corners coordinates of aruco markers in tuple
    number_of_aruco_markers = len(ids)  # the number of aruco markers, which were detected on image
    corners = np.concatenate(corners)  # transform corners ndarray to array
    ids = np.concatenate(ids)  # transform ids ndarray to array
    start_index = 0
    for i in range(number_of_aruco_markers):
        if ids[i] % 2 == 0:
            start_index = i
            break
    start_aruco_id = ids[start_index]  # id of starting aruco marker
    start_corner_coordinates = corners[start_index]  # coordinates of corners of starting aruco marker
    height, width, channels = img.shape
    x_center_img = int(width / 2)
    y_center_img = int(height / 2)
    node_1 = start_corner_coordinates[0]
    node_2 = start_corner_coordinates[1]
    node_3 = start_corner_coordinates[2]
    node_4 = start_corner_coordinates[3]
    index_list = [0, 1, 2, 3]
    sum_list = [np.sum(node_1), np.sum(node_2), np.sum(node_3), np.sum(node_4)]
    max_sum = max(sum_list)
    index_max = sum_list.index(max_sum)
    min_sum = min(sum_list)
    index_min = sum_list.index(min_sum)
    (x_bottom_right, y_bottom_right) = start_corner_coordinates[index_max]
    (x_top_left, y_top_left) = start_corner_coordinates[index_min]
    index_list.remove(index_min)
    index_list.remove(index_max)
    if start_corner_coordinates[index_list[0]][1] > start_corner_coordinates[index_list[1]][1]:
        (x_bottom_left, y_bottom_left) = start_corner_coordinates[index_list[0]]
        (x_top_right, y_top_right) = start_corner_coordinates[index_list[1]]
    else:
        (x_bottom_left, y_bottom_left) = start_corner_coordinates[index_list[1]]
        (x_top_right, y_top_right) = start_corner_coordinates[index_list[0]]
    aruco_dx = x_top_right - x_top_left
    aruco_dy = y_bottom_right - y_top_right
    x_center_aruco = (x_top_right + x_top_left) / 2
    y_center_aruco = (y_top_right + y_bottom_right) / 2
    crop_parameters = np.array([])
    scale_1 = 19
    scale_2 = 9
    # consider four events
    if start_aruco_id % 2 == 0:  # in our problem aruco markers with even ids situated in corners of square
        if x_center_aruco < x_center_img and y_center_aruco < y_center_img:  # left top marker
            print('left top marker was chosen')
            crop_parameters = np.append(crop_parameters, x_top_left)
            crop_parameters = np.append(crop_parameters, x_top_left + scale_1 * aruco_dx)
            crop_parameters = np.append(crop_parameters, y_top_left)
            crop_parameters = np.append(crop_parameters, y_top_left + scale_1 * aruco_dy)
        elif x_center_aruco < x_center_img and y_center_aruco > y_center_img:  # left bottom marker
            print('left bottom marker was chosen')
            crop_parameters = np.append(crop_parameters, x_bottom_left)
            crop_parameters = np.append(crop_parameters, x_bottom_left + scale_1 * aruco_dx)
            crop_parameters = np.append(crop_parameters, y_bottom_left - scale_1 * aruco_dy)
            crop_parameters = np.append(crop_parameters, y_bottom_left)
        elif x_center_aruco > x_center_img and y_center_aruco > y_center_img:  # right bottom marker
            print('right bottom marker was chosen')
            crop_parameters = np.append(crop_parameters, x_bottom_right - scale_1 * aruco_dx)
            crop_parameters = np.append(crop_parameters, x_bottom_right)
            crop_parameters = np.append(crop_parameters, y_bottom_right - scale_1 * aruco_dy)
            crop_parameters = np.append(crop_parameters, y_bottom_right)
        elif x_center_aruco > x_center_img and y_center_aruco < y_center_img:  # right top marker
            print('right top marker was chosen')
            crop_parameters = np.append(crop_parameters, x_top_right - scale_1 * aruco_dx)
            crop_parameters = np.append(crop_parameters, x_top_right)
            crop_parameters = np.append(crop_parameters, y_top_right)
            crop_parameters = np.append(crop_parameters, y_top_right + scale_1 * aruco_dy)
    else:  # there doesn't exist aruco in corners of square
        if x_center_aruco < x_center_img and y_center_img / 2 < y_center_aruco < 3 * y_center_img / 2:
            print('left middle aruco was chosen')
            crop_parameters = np.append(crop_parameters, x_top_left)
            crop_parameters = np.append(crop_parameters, x_top_left + scale_1 * aruco_dx)
            crop_parameters = np.append(crop_parameters, y_top_left - scale_2 * aruco_dy)
            crop_parameters = np.append(crop_parameters, y_bottom_left + scale_2 * aruco_dy)
        if x_center_aruco > x_center_img and y_center_img / 2 < y_center_aruco < 3 * y_center_img / 2:
            print('right middle aruco was chosen')
            crop_parameters = np.append(crop_parameters, x_top_right - scale_1 * aruco_dx)
            crop_parameters = np.append(crop_parameters, x_top_right)
            crop_parameters = np.append(crop_parameters, y_top_left - scale_2 * aruco_dy)
            crop_parameters = np.append(crop_parameters, y_bottom_left + scale_2 * aruco_dy)
        if y_center_aruco > y_center_img and x_center_img / 2 < x_center_aruco < 3 * x_center_img / 2:
            print('bottom middle aruco was chosen')
            crop_parameters = np.append(crop_parameters, x_bottom_left - scale_2 * aruco_dx)
            crop_parameters = np.append(crop_parameters, x_bottom_right + scale_2 * aruco_dx)
            crop_parameters = np.append(crop_parameters, y_bottom_left - scale_1 * aruco_dy)
            crop_parameters = np.append(crop_parameters, y_bottom_left)
        if y_center_aruco < y_center_img and x_center_img / 2 < x_center_aruco < 3 * x_center_img / 2:
            print('top middle aruco was chosen')
            crop_parameters = np.append(crop_parameters, x_top_left - scale_2 * aruco_dx)
            crop_parameters = np.append(crop_parameters, x_top_right + scale_2 * aruco_dx)
            crop_parameters = np.append(crop_parameters, y_top_left)
            crop_parameters = np.append(crop_parameters, y_top_left + scale_1 * aruco_dy)
    return crop_parameters.astype(int)


def crop_with_fix_parameters(image, crop_parameters):
    crop_img = image[crop_parameters[2]:crop_parameters[3], crop_parameters[0]:crop_parameters[1]]
    return crop_img


def crop_image(img):
    aruco_dict = cv2.aruco.DICT_4X4_1000  # dictionary for our aruco markers
    aruco_dictionary = cv2.aruco.Dictionary_get(aruco_dict)  # dictionary
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


image_1 = cv2.imread('first2d_bad.jpg', cv2.IMREAD_COLOR)  # function imread is reading a picture
dictionary = cv2.aruco.DICT_4X4_1000  # dictionary for our aruco markers
crop_params = fix_aruco(image_1, dictionary)
cropped_image = crop_with_fix_parameters(image_1, crop_params)
cv2.imshow('Cropped image', cropped_image)
cv2.waitKey(0)


