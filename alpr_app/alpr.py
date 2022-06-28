
from IPython.display import Image
from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np
import pandas as pd
import os.path
import requests

from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

#input_path = r'test_dataset/okk7448.jpg'

#Image(input_path)

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def drawPred(classId, conf, left, top, right, bottom, frame):
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
    label = '%.2f' % conf
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255), cv2.FILLED)
    #cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        #print("out.shape : ", out.shape)
        for detection in out:
            #if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            #if scores[classId]>confThreshold:
            confidence = scores[classId]
            '''if detection[4]>confThreshold:
                print(detection[4], " - ", scores[classId], " - th : ", confThreshold)
                print(detection)'''
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    cropped=None
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        # calculate bottom and right
        bottom = top + height
        right = left + width
        
        #crop the plate out
        cropped = frame[top:bottom, left:right].copy()
        # drawPred
        drawPred(classIds[i], confidences[i], left, top, right, bottom, frame)
    if cropped is not None:
        return cropped

def find_contours(dimensions, img) :
    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    ii = cv2.imread('contour.jpg')
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44,24))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            #plt.imshow(ii, cmap='gray')
            #plt.title('Predict Segments')

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) # List that stores the character's binary image (unsorted)
            
    # Return characters on ascending order with respect to the x-coordinate (most-left character first)
            
    #plt.show()
    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])# stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res


def segment_characters(image) :

    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    """
    plt.imshow(img_binary_lp, cmap='gray')
    plt.title('Contour')
    plt.show()
    """
    cv2.imwrite('contour.jpg',img_binary_lp)

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list
loaded_model = Sequential()
loaded_model.add(Conv2D(16, (22,22), input_shape=(28, 28, 3), activation='relu', padding='same'))
loaded_model.add(Conv2D(32, (16,16), input_shape=(28, 28, 3), activation='relu', padding='same'))
loaded_model.add(Conv2D(64, (8,8), input_shape=(28, 28, 3), activation='relu', padding='same'))
loaded_model.add(Conv2D(64, (4,4), input_shape=(28, 28, 3), activation='relu', padding='same'))
loaded_model.add(MaxPooling2D(pool_size=(4, 4)))
loaded_model.add(Dropout(0.4))
loaded_model.add(Flatten())
loaded_model.add(Dense(128, activation='relu'))
loaded_model.add(Dense(36, activation='softmax'))
loaded_model.load_weights('C:/Users/jpedr/TCC/ALPR_final/alpr_app/checkpoints/my_checkpoint')
def fix_dimension(img): 
    new_img = np.zeros((28,28,3))
    for i in range(3):
        new_img[:,:,i] = img
        return new_img
  
def show_results():
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c
    output = []
    for i,ch in enumerate(char): #iterating over the characters
        img_ = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3) #preparing image for the model
        predict_x=loaded_model.predict(img)[0] 
        y_=np.argmax(predict_x)
        #y_ = loaded_model.predict_classes(img)[0] #predicting the class
        character = dic[y_]
        output.append(character) #storing the result in a list
        
    plate_number = ''.join(output)
    
    return plate_number

def put_text(image,position,text,size=16,color_text=(239, 255, 233)):
        im_pil = Image.fromarray(image)
        font = ImageFont.truetype("C:/Users/jpedr/TCC/ALPR_4.0/Services/src/alpr/font.ttf", size=size)
        draw = ImageDraw.Draw(im_pil)
        draw.text(position, text, fill=color_text, font=font)
        image = np.asarray(im_pil)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


def capturar():

    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    confThreshold = 0.5  
    nmsThreshold = 0.4
    inpWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))     
    inpHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))     

    classesFile = "C:/Users/jpedr/TCC/ALPR_final/alpr_app/yolo_utils/classes.names";
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    modelConfiguration = r"C:/Users/jpedr/TCC/ALPR_final/alpr_app/yolo_utils/darknet-yolov3.cfg";
    modelWeights = r"C:/Users/jpedr/TCC/ALPR_final/alpr_app/yolo_utils/lapi.weights";

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    crawl= None
    captura = None
    # RECONHECIMENTO DE PLACA
    while cv2.waitKey(1) < 0:

        hasFrame, frame = cap.read() #frame: an image object from cv2
        cv2.rectangle(frame,(0,0),(inpWidth,70),(32, 32, 32),-1)
        try:
            blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
            net.setInput(blob)
            outs = net.forward(getOutputsNames(net))
            cropped = postprocess(frame, outs)
            
            char=segment_characters(cropped)
            placa = show_results()
            print(placa)
            
            if  len(placa) == 7 and placa != crawl:
                try:
                    all_infos = []
                    url = 'http://127.0.0.1:5000/crawler'
                    myobj = {'placa': placa}
                    x = requests.post(url, json = myobj)
                    info = x.json()
                    #print(info)
                    if info:
                        all_infos.append(info)
                        plate_resource = info[-1]
                        venal = info[0].replace(".",",")
                        if venal == "Aliquota:": venal = "Não Estimado"
                        else: venal = "R$ " + info[0].replace(".",",")
                        marca = info[1] + ' '+info[2]+'/'+''+info[5]
                        cidade = info[10]+'/'+info[9]
                        headers = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"}
                        req = requests.get(plate_resource,headers=headers).content
                        plate = 'C:/Users/jpedr/TCC/ALPR_4.0/Services/src/alpr/plate.webp'
                        with open(plate, 'wb+') as f: f.write(req)
                        frame = put_text(frame, (175,5),'Modelo: '+marca)
                        frame = put_text(frame, (175,35),'Valor: '+venal)
                        frame = put_text(frame, (375,35),'Local: '+cidade)
                        plate = cv2.imread(plate)   
                        plate = cv2.resize(plate,(150,50))   
                        frame[10:10+50, 5:5+150] = plate
                        crawl = placa
                        captura=[marca,venal,cidade]
                except:
                    pass

        except:
            pass
        
        #roi[np.where(mask)] = 0
        if captura != None:
            plate = 'C:/Users/jpedr/TCC/ALPR_4.0/Services/src/alpr/plate.webp'
            frame = put_text(frame, (175,5),'Modelo: '+captura[0])
            frame = put_text(frame, (175,35),'Valor: '+captura[1])
            frame = put_text(frame, (375,35),'Local: '+captura[2])
            plate = cv2.imread(plate)   
            plate = cv2.resize(plate,(150,50))   
            frame[10:10+50, 5:5+150] = plate
        cv2.imshow("CAPTURA ALPR",frame)
        #return(frame)
        """
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break"""

capturar()





