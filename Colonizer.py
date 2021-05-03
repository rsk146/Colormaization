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
from sklearn import preprocessing

#NEW IMPROVED AGENT PLAN:
#KDTREE AND KNN FOR 15: SOFTMAX ON THOSE 15


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

def recolor(pic, centers, k):
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

def findLoss(pic):
    pic2 = img.imread("/Users/rsk146/Downloads/berry.png")
    pic2 *= 255
    pic2 = pic.astype(int)
    loss = 0
    for x in range(1, 255):
        for y in range(128, 255):
            loss += (np.linalg.norm(pic[x][y]-pic2[x][y]))
    return loss

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
    recolor(pic, centers, 5)
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
    plt.imshow(pic)
    plt.savefig("kNNberry.png", bbox_inches='tight', pad_inches=0)
    plt.show()
    print(findLoss(pic))


def get_avg_dist(pic, centers, k):
    total = 0
    for i in range(256):
        for j in range(256):
            total += dist(pic[i][j], centers[closest_center(centers, pic[i][j], k)])
    return total/65536.

def elbow_method(pic):
    x = [j for j in range(1, 11)]
    y =[]
    #make and write
    # with tqdm(total=10, position=0, leave=True) as pbar:
    #     for i in range(1,11):
    #         centers = kmeans(pic, i)
    #         y.append(get_avg_dist(pic, centers, i))
    #         pbar.update(1)
    # with open("elbow.txt", "w") as f:
    #     json.dump(x, f)
    #     json.dump(y, f)
    with open("elbow.txt", "r") as f:
        x = json.load(f)
    with open("elbow2.txt", "r") as g:
        y = json.load(g)
    y = 20*preprocessing.normalize([y])
    y = y[0]
    data = list(zip(x, y))
    data = np.array(data)
    plt.scatter(data[:,0], data[:,1])
    theta = -.25 + np.arctan2(data[:,1].max() - data[:,1].min(), data[:,0].max() - data[:,0].min())
    rot_matrix = np.array(((np.cos(theta), -1*np.sin(theta)), (np.sin(theta), np.cos(theta))))
    rot_data = np.array([np.dot(rot_matrix, d) for d in data])
    minY = np.min(rot_data[:,1])
    minInd = np.where(rot_data == minY)
    minPoint=rot_data[minInd[0][0]]
    theta = -1*theta
    rot_matrix = np.array(((np.cos(theta), -1*np.sin(theta)), (np.sin(theta), np.cos(theta))))
    finPoint = np.dot(rot_matrix, minPoint)
    print("Elbow Colors Value:",int(finPoint[0].round()))
    plt.scatter(rot_data[:,0], rot_data[:,1])
    plt.show()

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def sgd(pic, gray_vec, weight, eta, color):
    sim_count = 0
    loss = np.inf
    count = 0
    while(loss > 100):
        count+=1
        loss_prime = 0
        for x in range(1, 255):
            for y in range(1, 128):
                loss_prime+=(pic[x][y][color] - sigmoid(np.dot(weight, gray_vec[x][y])))**2
        if loss_prime > loss:
            sim_count +=1
        else:
            sim_count = 0
        if sim_count >=25:
            break
        loss = loss_prime
        #SGD
        #
        i = random.randint(1,254)
        j = random.randint(1,127)
        x_i = gray_vec[i][j]
        weight_prime = -eta*(pic[i][j][0]-sigmoid(np.dot(weight, x_i)))*(-sigmoid_prime(np.dot(weight, x_i)))*np.asarray(x_i) + weight
        weight = weight_prime
        if count >=5000:
            break
    return weight

def improved_agent(pic):
    pic = pic.astype(float)
    pic*=1/255
    grayPic = grayer(pic)
    gray_vec = [[[] for j in range(256)] for i in range(256)]
    #initialize inputs
    for x in range(1, 255):
        for y in range(1, 255):
            gray_vec[x][y] = neighbor_vector(grayPic, x, y)
    #training regimen
    #initialize weights
    weight_red, weight_green, weight_blue = [random.uniform(0, 1) for i in range(9)], [random.uniform(0, 1) for i in range(9)], [random.uniform(0, 1) for i in range(9)]
    #model = 255*sigmoid(np.dot(weight_red, gray_vec[x][y]))
    #learning rate
    eta = .05
    #loss and training
    weight_red, weight_green, weight_blue = sgd(pic, gray_vec, weight_red, eta, 0), sgd(pic, gray_vec, weight_green, eta, 1), sgd(pic, gray_vec, weight_blue, eta, 2)
    print(weight_red)
    print(weight_green)
    print(weight_blue)
    for x in range(1, 255):
        for y in range(128, 255):
            pic[x][y] = (sigmoid(np.dot(weight_red, gray_vec[x][y])), sigmoid(np.dot(weight_green, gray_vec[x][y])), sigmoid(np.dot(weight_blue, gray_vec[x][y])))
    plt.imshow(pic)
    plt.savefig("RegressionBerry.png", bbox_inches='tight', pad_inches=0)
    plt.show()

#elbow_method(pic)
#basicAgent(pic)
improved_agent(pic)  
