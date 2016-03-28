import sys
import os

sys.path.append("../../../python")

import caffe
import numpy as np
import matplotlib.pyplot as plt

net=caffe.Net("deploy.prototxt","_iter_10000.caffemodel",caffe.TEST)
image=caffe.io.load_image("../imageset_2/15.png",False)
image_rz=caffe.io.resize_image(image,(28,28))

net.blobs['data'].data[0][...]=image_rz.transpose((2,0,1))

net.forward()
for layername,blob in net.blobs.iteritems():
    if layername == 'prob':
        print blob.data
