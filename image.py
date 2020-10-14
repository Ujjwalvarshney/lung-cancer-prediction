# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 05:45:43 2020

@author: Admin
"""

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
model = load_model('model_vgg19.h5')
img = image.load_img('C:/Users/Admin/Desktop/lung disease/val/PNEUMONIA\person1946_bacteria_4874.jpeg',target_size = (224,224))
x= image.img_to_array(img)
x = np.expand_dims(x ,axis=0)
img_data = preprocess_input(x)
classes = model.predict(img_data)
