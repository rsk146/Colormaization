import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import time
import random
import math


#Grab file and resize to 255 range for project sake
pic = img.imread("/Users/rsk146/Downloads/berry.png")
pic *= 255
pic = pic.astype(int)

#show image
#plt.imshow(pic)
#plt.show()

def grayer(pic):
    R, G, B = pic[:,:, 0], pic[:,:, 1], pic[:, :, 2]
    grayPic = .21*R + .72*G + .07*B
    return grayPic

dist = lambda x, y: math.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2 + (x[2]-y[2])**2)

def closest_center(centers, point):
    minDist = dist(point, centers[0])
    index = 0
    for i in range(1, 5):
        currDist = dist(point, centers[i])
        if currDist < minDist:
            minDist = currDist
            index = i
    return index

def kmeans(pic):
    centers = []
    for i in range(5):
        centers.append([random.randint(0,255), random.randint(0,255), random.randint(0,255)])
    converge = False
    partitions = [[] for i in range(5)]
    while not converge:
        for x in range(256):
            for y in range(256):
                partitions[closest_center(centers, pic[x][y])].append(pic[x][y])
        new_centers = []
        i = 0
        for group in partitions:
            l = 1.0/len(group) if len(group) !=0 else 0
            if l== 0: 
                print('weird')
                group.append([random.randint(0,255), random.randint(0,255), random.randint(0,255)])
                l = 1.0    
            new_centers.append([sum(col) for col in zip(*group)])
            new_centers[-1] = [l*x for x in new_centers[-1]]
            i+=1
        new_centers = [[round(i) for i in j] for j in new_centers]
        converge = set(map(tuple, centers)) == set(map(tuple, new_centers))
        print(new_centers)
        centers = new_centers
    return centers, partitions
kmeans(pic)

#pic = np.reshape(pic, (256*256, 3))
