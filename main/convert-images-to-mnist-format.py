import os
from PIL import Image
import array
from array import *
from random import shuffle
from builtins import int
from cmath import inf


# Load from and save to
Names = [[r'C:\Users\I347798\git\project_tomato\test\TRAIN_DIR','train'], [r'C:\Users\I347798\git\project_tomato\test\VALID_DIR','test']]

for name in Names:

    data_image = array("B") 
    data_label = array("B")
#    print(name)
    print(data_image)
    FileList = []
    print("filelist =",  FileList)
    for dirname in os.listdir(name[0]): # [1:] Excludes .DS_Store from Mac OS
        print("dirname = ",dirname)
        path = os.path.join(name[0],dirname)
        print("path = ",path)
        print(FileList)
        for filename in os.listdir(path):
#            print("filename =", filename)
            if filename.endswith(".jpg"):
                FileList.append(os.path.join(name[0],dirname,filename))

    shuffle(FileList) # Usefull for further segmenting the validation set
#    print("filelist =",  FileList)
    
    for filename in FileList:
        
        label = int(filename.split('\\')[7])
#        print("lable = ", label)
        Im = Image.open(filename)
#        print("filename = ", filename)
#        print(Im)
        pixel = Im.load()
#        print("pixel = ", pixel)
        width, height = Im.size
#        print("wight height = ",width, height)
        for x in range(0,width):
#            print("x = ", x)
            for y in range(0,height):
#                print("y = ", y)
#                print("data_imag =", data_image)
#                print("111 =", data_image[0])
#                print(data_image.append(pixel[y,x]))
#                print("znach_pix_[x,y] =", pixel[y,x])
#                print("pixel =", [y,x])
#                pixel_list = list(pixel[y,x])
#                print("list_znach_pix_[x,y] =", pixel_list)
#                print(type(pixel_list))
#                print(type(pixel_list[1]))
#                print(type(data_image))
#                pixel_array = array("B", pixel_list)
#                print(pixel_array)
                data_image.extend(pixel[y,x])
#        print(data_label)        
        data_label.append(label) # labels start (one unsigned byte each)
    

    hexval = "{0:#0{1}x}".format(len(FileList),6) # number of files in HEX
    print("hexval = ", hexval)
    # header for label array

    header = array('B')
    header.extend([0,0,8,1,0,0])
    print("header = ", header)
    header.append(int('0x'+hexval[2:][:2],16))
    print("header = ", header)
    header.append(int('0x'+hexval[2:][2:],16))
    print("header = ", header)
    
    data_label = header + data_label

    # additional header for images array
   
#    if max([width,height]) <= 256:
#        header.extend([0,0,0,width,0,0,0,height])
#    else:
#        raise ValueError('Image exceeds maximum size: 256x256 pixels');

    header[3] = 3 # Changing MSB for image data (0x00000803)
    
    data_image = header + data_image
    
    output_file = open(name[1]+'-images-idx3-ubyte', 'wb')
    data_image.tofile(output_file)
    output_file.close()
    
    output_file = open(name[1]+'-labels-idx1-ubyte', 'wb')
    data_label.tofile(output_file)
    output_file.close()

# gzip resulting files

#for name in Names:
#    os.system('gzip '+name[1]+'-images-idx3-ubyte')
#    os.system('gzip '+name[1]+'-labels-idx1-ubyte')