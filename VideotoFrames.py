import imageio
import os, sys
import cv2
from scipy import ndimage
from semantic_segmentation import generate_segment
from move_target import move_target
import numpy as np
import shutil
import matplotlib.pyplot as plt

OUTPUT_DIR = './output/'
TEMP_DIR = './temp/'
FPS = 10
h, w = 64, 64
video = 'data/daman.mp4'
target = 'data/spiderman.png'

def get_frame(vidcap, sec):
    hasFrames, image = vidcap.read()
    return hasFrames, image

def preprocess_data(video_path, target_path):  
    # read the target image
    target_im = cv2.imread(target_path)
    
    # # Segment the foregound, background from the video and obtain the mask.
    target_im = ndimage.zoom(target_im, (float(400)/target_im.shape[0], float(400)/target_im.shape[1], 1), order=3, prefilter=False)        
    target_im, _, _ = generate_segment(target_im)

    # Read frames from video.
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    sec = 0
    frameRate = 0.3
    count = 0

    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    while success:
        sec = sec + frameRate
        sec = round(sec, 2)
        
        # Segment the foregound, background from the video and obtain the mask.
        image = ndimage.zoom(image, (float(400)/image.shape[0], float(400)/image.shape[1], 1), order=3, prefilter=False)        
        image_fg, image_bg, mask = generate_segment(image)
                
        count+=1
        print("Frame ", count, " completed.")
        cv2.imwrite(TEMP_DIR+'fg'+str(count)+'.png', image_fg)
        cv2.imwrite(TEMP_DIR+'bg'+str(count)+'.png', image_bg)
        plt.imshow(mask)
        plt.imsave(TEMP_DIR+'mask'+str(count)+'.png', mask)
        plt.close()
        success, image = get_frame(vidcap, sec)   
        if count == 20:
            break     
    vidcap.release()
    
    # Write the foreground of the source image and the shifted target image to the output directory.
    if os.path.exists(TEMP_DIR): 
        image = cv2.imread(TEMP_DIR+'fg1.png')
        
        # spiderman should move here
        target = move_target(image, target_im)

        image = ndimage.zoom(image, (float(h)/image.shape[0], float(w)/image.shape[1], 1), order=3, prefilter=False)
        target = ndimage.zoom(target, (float(h)/target.shape[0], float(w)/target.shape[1], 1), order=3, prefilter=False)

        for i in range(1, count):
            img = cv2.imread(TEMP_DIR+'fg'+str(i+1)+'.png')
            
            target_image = move_target(image, target_im)
            target_image = ndimage.zoom(target_image, (float(h)/target_image.shape[0], float(w)/target_image.shape[1], 1), order=3, prefilter=False)
            target = np.concatenate((target, target_image), axis=1)

            img = ndimage.zoom(img, (float(h)/img.shape[0], float(w)/img.shape[1], 1), order=3, prefilter=False)        
            image = np.concatenate((image, img), axis = 1)
        cv2.imwrite(OUTPUT_DIR+'fg.png', image)
        cv2.imwrite(OUTPUT_DIR+'target.png', target)
        
        image = cv2.imread(TEMP_DIR+'bg1.png')
        image = ndimage.zoom(image, (float(h)/image.shape[0], float(w)/image.shape[1], 1), order=3, prefilter=False)
        for i in range(1, count):
            img = cv2.imread(TEMP_DIR+'bg'+str(i+1)+'.png')
            img = ndimage.zoom(img, (float(h)/img.shape[0], float(w)/img.shape[1], 1), order=3, prefilter=False)        
            image = np.concatenate((image, img), axis = 1)
        cv2.imwrite(OUTPUT_DIR+'bg.png', image)

        image = cv2.imread(TEMP_DIR+'mask1.png')
        image = ndimage.zoom(image, (float(h)/image.shape[0], float(w)/image.shape[1], 1), order=3, prefilter=False)
        for i in range(1, count):
            img = cv2.imread(TEMP_DIR+'mask'+str(i+1)+'.png')
            img = ndimage.zoom(img, (float(h)/img.shape[0], float(w)/img.shape[1], 1), order=3, prefilter=False)        
            image = np.concatenate((image, img), axis = 1)
        cv2.imwrite(OUTPUT_DIR+'mask.png', image)

        # remove the temp subfolder after processing.
        shutil.rmtree(TEMP_DIR)


if __name__ == "__main__":
    preprocess_data(video, target)
