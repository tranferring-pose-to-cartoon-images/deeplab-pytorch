import cv2
import numpy as np

'''Move target image at the position in the frame where the source is.'''
def move_target(source_image, target_image):
    
    target_bw = cv2.threshold(cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1]
    source_bw = cv2.threshold(cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1]

    # select the points where the object is present
    pts_target = np.where(target_bw != 255)
    pts_source = np.where(source_bw != 255)  

    # find the boundary box points around the object
    bbox_target = np.min(pts_target[0]), np.max(pts_target[0]), np.min(pts_target[1]), np.max(pts_target[1])
    bbox_mask = np.min(pts_source[0]), np.max(pts_source[0]), np.min(pts_source[1]), np.max(pts_source[1])

    # move the target object to the source image location on Canvas
    canvas = np.zeros(source_image.shape, dtype=np.uint8)
    canvas.fill(255)

    target = target_image[bbox_target[0]:bbox_target[1], bbox_target[2]:bbox_target[3], :]
    mask = canvas[bbox_mask[0]:bbox_mask[1], bbox_mask[2]:bbox_mask[3], :]


    target_h, target_w, _ = target.shape
    mask_h, mask_w, _ = mask.shape
    fy = mask_h / target_h
    fx = mask_w / target_w
    scaled_target = cv2.resize(target, (0,0), fx=fx,fy=fy) # scale the target image 

    for i, row in enumerate(range(bbox_mask[0], bbox_mask[1])):
        for j, col in enumerate(range(bbox_mask[2], bbox_mask[3])):
                canvas[row, col, :] = scaled_target[i, j, :]         
    
    return canvas
