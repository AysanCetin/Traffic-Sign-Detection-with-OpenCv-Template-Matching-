import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from imutils.object_detection import non_max_suppression


###############################################################################
###############################################################################

def read(main_folder_path):
    
    main_folder_list = os.listdir(main_folder_path)
    
    test_set = []
    folders = []
    
    for i in main_folder_list:
        if ".ppm" in i:
            test_set.append(i)
        elif len(i) == 2:
            folders.append(i)
    
    training_set = []        
    for folder_name in folders:
        
        folder_path = main_folder_path + "/" + folder_name
        folder_list = os.listdir(folder_path)
        
        count = 0 
        for image_name in folder_list:
            image_path = folder_path + "/" + image_name
            image = plt.imread(image_path)
            training_set.append([image_path,image])
            count += 1
            if count > 10:
                break
    
    annotations = {}
    annt_path = main_folder_path + "/" + "gt.txt"
    with open(annt_path,"r") as annt:
        
        for line in annt:
            filename,x1,y1,x2,y2,label = line.split(";")
            
            if filename in annotations:
                annotations[filename].append([int(x1),int(y1),int(x2),int(y2)])
                
            else:
                annotations[filename] = [[int(x1),int(y1),int(x2),int(y2)]]
                
    return annotations, test_set, training_set

###############################################################################
###############################################################################

def detect(image,template,method):
        
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    h,w = gray_template.shape
    
    detect_threshold = cv2.matchTemplate(gray_image, gray_template, method)
    threshold = 0.9
    
    location = np.where(detect_threshold > threshold)
    
    results = []
    
    for i in zip(*location[::-1]):
        results.append([i[0],i[1],i[0]+w,i[1]+h])
        
    return results

###############################################################################
###############################################################################
    
main_folder_path = "/home/artint/Downloads/FullIJCNN2013"

annotation, dataSet, trainingSet = read(main_folder_path)
print(len(trainingSet))

f = open("template_detects.txt","w+")

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

method = eval(methods[1])


for i in range(len(dataSet)):
    
    start = timer()
    
    image = cv2.imread(main_folder_path + "/" + dataSet[i])
    print(dataSet[i])

    detects = []
    
    for name,temp in trainingSet:
        
        results = detect(image.copy(), temp, method)
        
        if (len(results)>0):
            detects.extend(results)
    
    array_detects = np.array([[x1,y1,x2,y2] for (x1,y1,x2,y2) in detects])
    detects_rects = non_max_suppression(array_detects, probs = None, overlapThresh = 0.2)
    
    for x1,y1,x2,y2 in detects_rects:
        cv2.rectangle(image, (x1,y1),(x2,y2),(255,255,0),3)
        f.write(dataSet[i] + ";" + str(x1) + ";" + str(y1) + ";" + str(x2) + ";" + str(y2) + "\n")
    
    cv2.imshow("Detection Result ", cv2.resize(image,(448,448)))
    
    end = timer()
    print("elapsed time:", end-start)
    print(detects_rects)
    
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

    if (len(detects) > 0):
        cv2.waitKey(0)
        
f.close
