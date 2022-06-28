
from IPython.display import Image
from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np
import pandas as pd
import requests
import time

#input_path = r'test_dataset/okk7448.jpg'

#Image(input_path)
cap = cv2.VideoCapture(0)
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
       
        for detection in out:

            scores = detection[5:]
            classId = np.argmax(scores)
          
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
        

        bottom = top + height
        right = left + width
        
  
        cropped = frame[top:bottom, left:right].copy()

        drawPred(classIds[i], confidences[i], left, top, right, bottom, frame)
    if cropped is not None:
        return cropped

def put_text(image,position,text,size=16,color_text=(239, 255, 233)):
    im_pil = Image.fromarray(image)
    font = ImageFont.truetype("C:/Users/jpedr/TCC/ALPR_final/alpr_app/font.ttf", size=size)
    draw = ImageDraw.Draw(im_pil)
    draw.text(position, text, fill=color_text, font=font)
    image = np.asarray(im_pil)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# RECONHECIMENTO DE PLACA
class camera:

    crawl= None
    captura = None
    print('iniciando camera')
    while cv2.waitKey(1) < 0:

        hasFrame, frame = cap.read() #frame: an image object from cv2
        blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))
        cropped = postprocess(frame, outs)
        cv2.rectangle(frame,(0,0),(inpWidth,70),(32, 32, 32),-1)
        if cropped is not None:
            cv2.imwrite('C:/Users/jpedr/TCC/ALPR_final/alpr_app/cropped.jpg',cropped)
            url = 'http://127.0.0.1:5000/ocr'
            x = requests.post(url)
            placa=x.json()
            print(placa)
            frame = put_text(frame, (inpWidth -100,inpHeight -30),placa)
            if  len(placa) == 7 and placa != crawl:
                try:
                    all_infos = []
                    url = 'http://127.0.0.1:5000/crawler'
                    myobj = {'placa': placa}
                    x = requests.post(url, json = myobj)
                    info = x.json()
                    #print(info)A
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
                        plate = 'C:/Users/jpedr/TCC/ALPR_final/alpr_app/plate.webp'
                        with open(plate, 'wb+') as f: f.write(req)
                        frame = put_text(frame, (175,5),'Modelo: '+marca)
                        frame = put_text(frame, (175,35),'Valor: '+venal)
                        frame = put_text(frame, (375,35),'Local: '+cidade)
                        try:
                            plate = cv2.imread(plate)   
                            plate = cv2.resize(plate,(150,50))  
                            frame[10:10+50, 5:5+150] = plate
                        except:
                            pass 
                        crawl = placa
                        captura=[marca,venal,cidade]
                except Exception as e:
                    print(str(e))

        #roi[np.where(mask)] = 0
        if captura != None:
            plate = 'C:/Users/jpedr/TCC/ALPR_final/alpr_app/plate.webp'
            frame = put_text(frame, (175,5),'Modelo: '+captura[0])
            frame = put_text(frame, (175,35),'Valor: '+captura[1])
            frame = put_text(frame, (375,35),'Local: '+captura[2])
            plate = cv2.imread(plate)   
            plate = cv2.resize(plate,(150,50))   
            frame[10:10+50, 5:5+150] = plate
        frames = frame
        #return frames
        #cv2.imshow("CAPTURA ALPR",frame)
        #time.sleep(1)


#cam()



