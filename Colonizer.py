import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import time
import random
import math
import itertools
from itertools import chain
import json
from tqdm import tqdm
#whole test right now relies on 256x256 test image
#Grab file and resize to 255 range for project sake
pic = img.imread("/Users/rsk146/Downloads/berry.png")
pic *= 255
pic = pic.astype(int)
#reshape tool?
#pic = np.reshape(pic, (256*256, 3))

#grayscale
#plt.imshow(pic, cmap = "gray")

#greyvec?
#gray_vec_list = [elem for twod in gray_vec for elem in twod]

#scikit kdtree and take the resulting index for 6 nearest neighbors
#get it back with [int(x/256)][x%256]

def show_image(pic):
    plt.imshow(pic)
    plt.show()

def grayer(pic):
    R, G, B = pic[:,:, 0], pic[:,:, 1], pic[:, :, 2]
    grayPic = .21*R + .72*G + .07*B
    return grayPic

dist = lambda x, y: math.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2 + (x[2]-y[2])**2)

def get_eu_dist(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point1 - point2)

def closest_center(centers, point, k):
    minDist = dist(point, centers[0])
    index = 0
    for i in range(1, k):
        currDist = dist(point, centers[i])
        if currDist < minDist:
            minDist = currDist
            index = i
    return index

def kmeans(pic, k):
    centers = []
    for i in range(k):
        centers.append([random.randint(0,255), random.randint(0,255), random.randint(0,255)])
    converge = False
    partitions = [[] for i in range(k)]
    while not converge:
        for x in range(256):
            for y in range(256):
                partitions[closest_center(centers, pic[x][y], k)].append(pic[x][y])
        new_centers = []
        i = 0
        for group in partitions:
            l = 1.0/len(group) if len(group) !=0 else 0
            if l== 0: 
                #print('weird')
                #print(i)
                group.append([random.randint(0,255), random.randint(0,255), random.randint(0,255)])
                l = 1.0    
            new_centers.append([sum(col) for col in zip(*group)])
            new_centers[-1] = [l*x for x in new_centers[-1]]
            i+=1
        new_centers = [[round(i) for i in j] for j in new_centers]
        converge = set(map(tuple, centers)) == set(map(tuple, new_centers))
        #print(new_centers)
        centers = new_centers
    return centers

def recolor(pic, centers):
    for x in range(256):
        for y in range(0, 128):
            pic[x][y] = centers[closest_center(centers, pic[x][y], k)]            

#256^2=65536 pix
#bordered =  256^2 - 256*4-4 =64516 pix

def neighbor_vector(grayPic, x, y):
    return [round(grayPic[i][j], 2) for i in range(x-1, x+2) for j in range(y-1, y+2)]

def find_index(outVal, d):
    for i in range(5):
        if d < outVal[i]:
            return i
    return 5

def nearest_neighbors(x, y, gray_vec):
    out = [(1,1), (1,2), (1, 3), (1,4), (1,5), (1,6)]
    outVal = []
    for point in out:
        outVal.append(get_eu_dist(gray_vec[x][y], gray_vec[point[0]][point[1]]))
    zipped = list(zip(outVal, out))
    sorted_zipped = sorted(zipped, key= lambda x: x[0])
    outVal, out = zip(*sorted_zipped)
    outVal = list(outVal)
    out = list(out)
    for i in range(1, 255):
        for j in range(7, 128):
            d = get_eu_dist(gray_vec[i][j], gray_vec[x][y])
            if d < outVal[5]:
                ind = find_index(outVal, d)
                outVal.insert(ind, d)
                out.insert(ind, (i,j))
                outVal.pop()
                out.pop()
    return out

def findMaj(colorVal):
    m = {}
    for i in range(6):
        if tuple(colorVal[i]) in m:
            m[tuple(colorVal[i])]+=1
        else:
            m[tuple(colorVal[i])] = 1
    c = 0
    for key in m:
        if m[key] > 3:
            return key
    return colorVal[0]
        
def basicAgent(pic):
    grayPic = grayer(pic)
    #run kmeans
    # centers = kmeans(pic, 5)
    # with open("kmeans.txt", "w") as f:
    #     json.dump(centers, f)
    # print("wrote kmeans")

    #use preran kmeans
    centers = []
    with open("kmeans.txt", "r") as f:
        centers = json.load(f)
    centers = list(centers)
    print("Read kmeans centers")
    print(centers)
    recolor(pic, centers)
    #show_image(pic)
    # plt.imshow(grayPic, cmap = "gray")
    # plt.show()
    gray_vec = [[[] for j in range(256)] for i in range(256)]
    for x in range(1, 255):
        for y in range(1, 255):
            gray_vec[x][y] = neighbor_vector(grayPic, x, y)
    #run nearest neighbors on the right half
    print("gray vec created")
    with tqdm(total=127*255, position=0, leave=True) as pbar:
        for x in range(1, 255):
            for y in range(128, 255):
                #print(x,y)
                nn = nearest_neighbors(x, y, gray_vec)
                colorVal = []
                for point in nn:
                    i, j = point[0], point[1]
                    colorVal.append(pic[i][j])
                pic[x][y] = findMaj(colorVal)
                pbar.update(1)
    #should dump pic data here lol retard
    show_image(pic)

def get_avg_dist(pic, centers, k):
    total = 0
    for i in range(256):
        for j in range(256):
            total += dist(pic[i][j], centers[closest_center(centers, pic[i][j], k)])
    return total/65536.


def elbow_method(pic):
    x = [j for j in range(1, 11)]
    y =[]
    with tqdm(total=10, position=0, leave=True) as pbar:
        for i in range(1,11):
            centers = kmeans(pic, i)
            y.append(get_avg_dist(pic, centers, i))
            pbar.update(1)
    with open("elbow.txt", "w") as f:
        json.dump(x, f)
        json.dump(y, f)
    data = list(zip(x, y))
    data = np.array(data)
    plt.scatter(data[:,0], data[:,1])
    plt.show()
    theta = np.arctan2(data[:,1].max() - data[:,1].min(), data[:,0].max() - data[:,0].min())
    rot_matrix = np.array(((np.cos(theta), -1*np.sin(theta)), (np.sin(theta), np.cos(theta))))
    rot_data = data.dot(rot_matrix)
    plt.scatter(rot_data[:,0], rot_data[:,1])
    plt.show()
    print(np.where(rot_data == rot_data.min())[0][0])

# def improved_agent(pic):
#     grayPic = grayer(pic)
#     gray_vec = [[[] for j in range(256)] for i in range(256)]
#     for x in range(1, 255):
#         for y in range(1, 255):
#             gray_vec[x][y] = neighbor_vector(grayPic, x, y)
    


elbow_method(pic)
#basicAgent(pic)



