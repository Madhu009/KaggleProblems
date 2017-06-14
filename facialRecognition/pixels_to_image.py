#this program converts pixel to images

# coding: utf-8

'''
This script creates 3-channel gray images from FER 2013 dataset.
It has been done so that the CNNs designed for RGB images can
be used without modifying the input shape.

This script requires two command line parameters:
1. The path to the CSV file
2. The output directory

It generates the images and saves them in three directories inside
the output directory - Training, PublicTest, and PrivateTest.
These are the three original splits in the dataset.
'''


import os
import csv
import numpy as np
import scipy.misc


filepath="C:/Users/Madhu/Desktop/kag/tr.csv"
dirpath="C:/Users/Madhu/Desktop/kag"

w,h=48,48
image=np.zeros((h,w),dtype=np.uint8)
id=1
with open(filepath,"r") as csvfile:
    data=csv.reader(csvfile,delimiter=",")
    headers=next(data)
    print(headers)

    for row in data:
        emotion=row[0]
        pixels=list(map(int,row[1].split()))
        usage=row[2]

        pixels_array = np.array(pixels)

        image=pixels_array.reshape(w,h)

        stacked_image = np.dstack((image,) * 3)

        image_folder = os.path.join(dirpath, usage)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        image_file = os.path.join(image_folder, str(id) + '.jpg')
        scipy.misc.imsave(image_file, stacked_image)
        id += 1
        if id % 100 == 0:
            print('Processed {} images'.format(id))

print("Finished processing {} images".format(id))

