"""
visualize results for test image
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

from module import transforms as transforms
from skimage import io
from skimage.transform import resize
from module import models
#from models import *

cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def net(raw_img,netpath):
    #raw_img = io.imread(imgpath)
    gray = rgb2gray(raw_img)
    gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)

    img = gray[:, :, np.newaxis]

    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    net = models.VGG('VGG19')
    #=========================================================
    '''
    print (net)
    params = net.state_dict()
    print( type(params))
    for k, v in params.items():
        print(k)
    print(params['features.49.bias'])
    print(params['features.49.bias'].shape)
    '''
    #=========================================================
    checkpoint = torch.load(os.path.join(netpath), map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    # net.cuda()
    net.eval()
    ncrops, c, h, w = np.shape(inputs)
    # =========================================================
    '''
    print("ncrops"+str(ncrops))
    print("c"+str(c))
    print("h"+str(h))
    print("w"+str(w))
    '''
    # =========================================================
    inputs = inputs.view(-1, c, h, w)
    # inputs = inputs.cuda()
    inputs = Variable(inputs, volatile=True)
    outputs = net(inputs)
    # =========================================================
    '''
    print (outputs)
    print(outputs.shape)
    '''
    # =========================================================
    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
    # =========================================================
    #print(outputs_avg)
    # =========================================================
    '''
    score = F.softmax(outputs_avg)
    _, predicted = torch.max(outputs_avg.data, 0)

    #print("The Expression is %s" % str(class_names[int(predicted.cpu().numpy())]))

    result = int(predicted.cpu().numpy())
    result1=str(class_names[int(predicted.cpu().numpy())])
    return result,result1
    '''
    return outputs_avg

def net7(raw_img,netpath):
    #raw_img = io.imread(imgpath)
    gray = rgb2gray(raw_img)
    gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)

    img = gray[:, :, np.newaxis]

    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    net = models.VGG7('VGG19')
    #=========================================================
    '''
    print (net)
    params = net.state_dict()
    print( type(params))
    for k, v in params.items():
        print(k)
    print(params['features.49.bias'])
    print(params['features.49.bias'].shape)
    '''
    #=========================================================
    checkpoint = torch.load(os.path.join(netpath), map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    # net.cuda()
    net.eval()
    ncrops, c, h, w = np.shape(inputs)
    # =========================================================
    '''
    print("ncrops"+str(ncrops))
    print("c"+str(c))
    print("h"+str(h))
    print("w"+str(w))
    '''
    # =========================================================
    inputs = inputs.view(-1, c, h, w)
    # inputs = inputs.cuda()
    inputs = Variable(inputs, volatile=True)
    outputs = net(inputs)
    # =========================================================
    '''
    print (outputs)
    print(outputs.shape)
    '''
    # =========================================================
    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
    # =========================================================
    #print(outputs_avg)
    # =========================================================

    score = F.softmax(outputs_avg)
    _, predicted = torch.max(outputs_avg.data, 0)

    #print("The Expression is %s" % str(class_names[int(predicted.cpu().numpy())]))

    result = int(predicted.cpu().numpy())
    result1=str(class_names[int(predicted.cpu().numpy())])
    return result,result1

