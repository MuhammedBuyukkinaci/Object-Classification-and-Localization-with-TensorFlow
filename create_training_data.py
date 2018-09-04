#Before running this script, all photos were resized to (227,227,3). Then, All photos were labeled via LabelImg.

# Importing Dependencies
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from random import shuffle

#Setting directories
MAIN_DIR = os.getcwd()
working_dir = os.path.join(MAIN_DIR,"training_images")


os.chdir(working_dir)

## Mirroring with respect to y axis

# Defining a mirroring function for flipping wrt y
def mirroring_y(data):
    list_of_lists = [['xmin',19],['xmax',21]]
    min_max_list = []
    for v in list_of_lists:
        sep_1 = v[0]
        a = data[v[1]].split(sep_1)[0]
        b = data[v[1]].split(sep_1)[1]
        c = data[v[1]].split(sep_1)[2]
        sep_2 = '<'
        b_1 = b.split(sep_2)[0]
        b_2 = b.split(sep_2)[1]
        b_1_1 = '>'
        b_1_2 = str(227 - int(b_1[1:])) 
        changed_string = a + sep_1 + b_1_1 + b_1_2 + sep_2 + b_2 + sep_1 + c
        min_max_list.append(changed_string)
    return list((min_max_list))

def make_flipping_y():
    for i in tqdm([x for x in os.listdir() if x.split('.')[1] == 'jpg' ]):
        vegetable_name = i.split('.')[0]
        path = os.path.join(working_dir,i) # Setting the directory of image which is going to be read.
        img = cv2.imread(path,1) # Reading image
        img = cv2.flip( img, 1 ) # Mirroring image wrt y axis
        cv2.imwrite(vegetable_name + '_y' + '.jpg',img) # Saving flipped image
        with open(str(vegetable_name + '.xml')) as f: # Reading the .xml file of image
            content = f.readlines()
        [content[19],content[21]] = mirroring_y(content)    
        with open(vegetable_name + '_y' + '.xml', "w") as f:
            for s in content:
                f.write(str(s))
                
make_flipping_y()


## Mirroring with respect to x axis

# Defining a mirroring function for flipping wrt x
def mirroring_x(data):
    list_of_lists = [['ymin',20],['ymax',22]]
    min_max_list = []
    for v in list_of_lists:
        sep_1 = v[0]
        a = data[v[1]].split(sep_1)[0]
        b = data[v[1]].split(sep_1)[1]
        c = data[v[1]].split(sep_1)[2]
        sep_2 = '<'
        b_1 = b.split(sep_2)[0]
        b_2 = b.split(sep_2)[1]
        b_1_1 = '>'
        b_1_2 = str(227 - int(b_1[1:])) 
        changed_string = a + sep_1 + b_1_1 + b_1_2 + sep_2 + b_2 + sep_1 + c
        min_max_list.append(changed_string)
    return list(min_max_list)  


def make_flipping_x():
    for i in tqdm([x for x in os.listdir() if x.split('.')[1] == 'jpg' ]):
        vegetable_name = i.split('.')[0]
        path = os.path.join(working_dir,i)# Setting the directory of image which is going to be read.
        img = cv2.imread(path,1) # Reading image
        img = cv2.flip( img, 0 ) # Mirroring image wrt y axis
        cv2.imwrite(vegetable_name + '_x' + '.jpg',img) # Saving flipped image
        with open(str(vegetable_name + '.xml')) as f: # Reading the .xml file of image
            content = f.readlines()
        [content[20],content[22]] = mirroring_x(content)    
        with open(vegetable_name + '_x' + '.xml', "w") as f:
            for s in content:
                f.write(str(s))

make_flipping_x()


# Adding noise for all images
def add_noise():
    for i in tqdm([x for x in os.listdir() if x.split('.')[1] == 'jpg' ]):
        vegetable_name = i.split('.')[0]
        path = os.path.join(working_dir,i)
        img = cv2.imread(path,1)
        noise  = 2 * img.std() * np.random.random(img.shape)
        img = img + noise
        cv2.imwrite(vegetable_name + '_noised' + '.jpg',img)
        with open(str(vegetable_name + '.xml')) as f:
            content = f.readlines()
        with open(vegetable_name + '_noised' + '.xml', "w") as f:
            for s in content:
                f.write(str(s))

add_noise()


def xml_reader(data):
    list_of_lists = [['xmin',19],['ymin',20],['xmax',21],['ymax',22]]
    min_max_list = []
    for v in list_of_lists:
        sep_1 = v[0]
        a = data[v[1]].split(sep_1)[0]
        b = data[v[1]].split(sep_1)[1]
        c = data[v[1]].split(sep_1)[2]
        sep_2 = '<'
        b_1 = b.split(sep_2)[0]
        b_2 = b.split(sep_2)[1]
        b_1_1 = '>'
        b_1_2 =  int(b_1[1:])
        val = b_1_2
        min_max_list.append(val)
    if min_max_list[0] > min_max_list[2]:
        min_max_list[0], min_max_list[2] = min_max_list[2], min_max_list[0]
    if min_max_list[1] > min_max_list[3]:
        min_max_list[1], min_max_list[3] = min_max_list[3], min_max_list[1]
    return min_max_list


def create_data():
    training_data = []
    for i in tqdm([x for x in os.listdir() if x.split('.')[1] == 'jpg' ]):
        vegetable_name = i.split('.')[0]
        vegetable_name_updated = vegetable_name.split('_')[0]
        if vegetable_name_updated =='cucumber':
            output_vector = [1,0,0]
        elif vegetable_name_updated =='eggplant':
            output_vector = [0,1,0]
        else:
            output_vector = [0,0,1]
        path = os.path.join(working_dir,i)
        img = cv2.imread(path,1)
        with open(str(vegetable_name + '.xml')) as f:
            content = f.readlines()
        locations = xml_reader(content)
        training_data.append([np.array(img),np.array(output_vector),np.array(locations)])
    shuffle(training_data)
    return training_data

data = create_data()
os.chdir(MAIN_DIR)
np.save('object_localization.npy', data)





