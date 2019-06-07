# Handle the case where the person moves out of the frame

import os
import tarfile
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import scipy.io as scpio
from PIL import Image
from skimage.transform import resize
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from scipy import ndimage
import pdb

class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME =  'ResizeBilinear_3:0'  #'SemanticPredictions:0'

    INPUT_SIZE = 321
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
      """Creates and loads pretrained deeplab model."""
      self.graph = tf.Graph()

      graph_def = None
      # Extract frozen graph from tar archive.
      tar_file = tarfile.open(tarball_path)
      for tar_info in tar_file.getmembers():
        if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
          file_handle = tar_file.extractfile(tar_info)
          graph_def = tf.GraphDef.FromString(file_handle.read())
          break

      tar_file.close()

      if graph_def is None:
        raise RuntimeError('Cannot find inference graph in tar archive.')

      with self.graph.as_default():
          tf.import_graph_def(graph_def, name='')

      logits = self.graph.get_tensor_by_name(self.OUTPUT_TENSOR_NAME)
      self.softmax_output = tf.nn.softmax(logits)

      self.sess = tf.Session(graph=self.graph)
      

    def run(self, image):
      """Runs inference on a single image.

      Args:
        image: A PIL.Image object, raw input image.

      Returns:
        resized_image: RGB image resized from original input image.
        seg_map: Segmentation map of `resized_image`.
      """
      width, height = image.size      
      
      resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
      target_image_size = (int(resize_ratio * width), int(resize_ratio * height))

      resized_image = image.convert('RGB').resize(target_image_size, Image.ANTIALIAS)

      batch_seg_map = self.sess.run(
          self.softmax_output,
          feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(image)]})
      
      seg_map = batch_seg_map[0]

      sliced_seg_map = seg_map[0:height, 0:width, :]

      return image, sliced_seg_map[:, :, 15] # return only the person class



def generate_segment(image):

    MODEL = DeepLabModel('./model/deeplab_model.tar.gz')

    # Obtain the segmentation mask for the person class.    
    try:
        image = Image.fromarray(image)

        resized_im, seg_map = MODEL.run(image)
        
        plt.figure()
        plt.imshow(seg_map)
        plt.imsave('output.png', seg_map)
        plt.axis("off")

    except Exception as e:
        print("Some exception occured while running the model on a frame.")
        print('Exception: ', e)
        exit()
  
    try:
        seg_map = cv2.imread('output.png') # load the segmentation map to obtain RGB image
        im_gray = cv2.cvtColor(seg_map, cv2.COLOR_BGR2GRAY) 
        
        # separate the foreground from the image
        im_bw = cv2.threshold(im_gray, 127, 255, cv2.THRESH_BINARY)[1]  
        mask = im_bw == 255 
        image = np.array(image)
        image_fg = image.copy()
        image_bg = image.copy()

        image_fg[mask==False] = -1 
        image_bg[mask==True] = 255
        
        return image_fg, image_bg, mask

    except Exception as e:
        print("Exception: ", e)
        exit()
