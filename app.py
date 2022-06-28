from flask import Flask, render_template, Response, request, abort,flash,redirect,url_for
import cv2
#from alpr_app.Cropped import camera
from alpr_app.OCR import ocr
import json
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from bs4 import BeautifulSoup as bs
import numpy as np
from IPython.display import Image
from PIL import Image, ImageFont, ImageDraw
import requests
import datetime
import sqlite3 as sql
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
import os

diretorio = os.getcwd()
app = Flask(__name__,static_folder=diretorio+'/static')

cap = cv2.VideoCapture(0)
confThreshold = 0.5  
nmsThreshold = 0.4
inpWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))     
inpHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))     
inpWidth = 416    
inpHeight = 416    

classesFile = diretorio+"/alpr_app/yolo_utils/classes.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

modelConfiguration = diretorio + r"/alpr_app/yolo_utils/darknet-yolov3.cfg";
modelWeights = diretorio + r"/alpr_app/yolo_utils/lapi.weights";

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def getOutputsNames(net):
    layersNames = net.getLayerNames()
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

def put_text(image,position,text,color_text,size=16):
    im_pil = Image.fromarray(image)
    font = ImageFont.truetype(diretorio+"/alpr_app/font.ttf", size=size)
    draw = ImageDraw.Draw(im_pil)
    draw.text(position, text, fill=color_text, font=font)
    image = np.asarray(im_pil)
    return image


def gen_frames():
    crawl= None
    captura = None
    print('iniciando camera')
    while cv2.waitKey(1) < 0:

        hasFrame, frame = cap.read() 
        blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))
        cropped = postprocess(frame, outs)
        cv2.rectangle(frame,(0,0),(inpWidth,70),(32, 32, 32),-1)
        if cropped is not None:
            cv2.imwrite(diretorio+'/alpr_app/cropped.jpg',cropped)
            url = 'http://127.0.0.1:5000/ocr'
            x = requests.post(url)
            placa=x.json()
            #print(placa)
            frame = put_text(frame, (inpWidth -100,inpHeight -30),placa,(153,0,0))
            if len(placa)<7:
                placa = ocr(cropped)
                #time.sleep(1)
            else:
                pass
            print(placa)
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
                        plate = diretorio+'/alpr_app/plate.webp'
                        with open(plate, 'wb+') as f: f.write(req)
                        frame = put_text(frame, (175,5),'Modelo: '+marca,(239, 255, 233))
                        frame = put_text(frame, (175,35),'Valor: '+venal,(239, 255, 233))
                        frame = put_text(frame, (375,35),'Local: '+cidade,(239, 255, 233))
                        try:
                            plate = cv2.imread(plate)   
                            plate = cv2.resize(plate,(150,50))  
                            frame[10:10+50, 5:5+150] = plate
                            agora = datetime.datetime.now()
                            con=sql.connect("banco_alpr.db")
                            cur=con.cursor()
                            cur.execute("insert into historico_veiculos(PLACA,MODELO,HORA_ENTRADA) values (?,?,?)",(placa,marca,agora))
                            con.commit()
                            flash('Entrada de veículo','success')
                        except Exception as error:
                            print(str(error)) 

                        crawl = placa
                        captura=[marca,venal,cidade]
                        
                except Exception as e:
                    print(str(e))

        #roi[np.where(mask)] = 0
        if captura != None:
            plate = diretorio+'/alpr_app/plate.webp'
            frame = put_text(frame, (175,5),'Modelo: '+captura[0],(239, 255, 233))
            frame = put_text(frame, (175,35),'Valor: '+captura[1],(239, 255, 233))
            frame = put_text(frame, (375,35),'Local: '+captura[2],(239, 255, 233))
            plate = cv2.imread(plate)   
            plate = cv2.resize(plate,(150,50))   
            frame[10:10+50, 5:5+150] = plate
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame

url = "https://placafipe.com/placa/"      
options = webdriver.FirefoxOptions()
options.add_argument('--headless')
driver = webdriver.Firefox(options=options)
executor_url  = driver.command_executor._url
session_id = driver.session_id
driver.get(url)
driver.implicitly_wait(30)

def find_contours(dimensions, img) :
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    ii = cv2.imread(diretorio+'/alpr_app/contour.jpg')
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) 

            char_copy = np.zeros((44,24))
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            char = cv2.subtract(255, char)
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) 
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])
    img_res = np.array(img_res_copy)

    return img_res


def segment_characters(image) :

    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
   
    cv2.imwrite(diretorio+'/alpr_app/contour.jpg',img_binary_lp)

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
loaded_model.load_weights(diretorio+'/alpr_app/checkpoints/my_checkpoint')


def fix_dimension(img): 
    new_img = np.zeros((28,28,3))
    for i in range(3):
        new_img[:,:,i] = img
        return new_img
  
def show_results(char):
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c
    output = []
    for i,ch in enumerate(char):
        img_ = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3) 
        predict_x=loaded_model.predict(img)[0] 
        y_=np.argmax(predict_x)
        character = dic[y_]
        output.append(character) 
        
    plate_number = ''.join(output)
    
    return plate_number


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
@app.route("/index")
def index():
    con=sql.connect("banco_alpr.db")
    #con.row_factory=sql.Row
    cur=con.cursor()
    cur.execute("select * from historico_veiculos")
    tabela=cur.fetchall()

    cur=con.cursor()
    cur.execute("select count(distinct PLACA) from historico_veiculos")
    cadastrados=cur.fetchone()

    cur=con.cursor()
    cur.execute("select count(PLACA) from historico_veiculos where HORA_SAIDA is null")
    ocupado=cur.fetchone()

    cur=con.cursor()
    cur.execute("select * from historico_veiculos where HORA_SAIDA is null")
    saida=cur.fetchall()

    return render_template('index.html',datas=tabela,cadastro=cadastrados[0],ocupa=ocupado[0],saidas=saida)




@app.route('/crawler', methods=['POST'])
def get_placa():
    if request.method == 'POST':
        print("placa: ", request.json)
        dados  = request.json
        placa = dados['placa']
        print(placa)
        def __get_info():
            infos_to_get = ["Marca:","Modelo:","Importado:","Ano:","Cor:","Cilindrada:",
                    "Potencia:","Combustível:","UF:","Município:"]
            infos_obtained = []
            for info in infos_to_get:
                try:
                    ret = tds_list[tds_list.index(info) + 1]  
                    infos_obtained.append(ret)
                except: infos_obtained.append(None)
            return infos_obtained
        try:
            driver.get(url + placa)
            driver.implicitly_wait(30)
            html =  driver.page_source
            cs = bs(html ,"lxml")
            #driver.close()
            tds_list = []
            tds = cs.findAll("td")
            
            for td in tds: tds_list.append(td.text)
            
            venal = cs.findAll("table")[1].findAll("td")[2].text.split(" ")[-1].replace(".","").replace(",",".")
            marca, modelo, importado, ano, cor, cilindrada, potencia, combustivel, uf, municipio = __get_info()
            modelo =  modelo.replace(",",".")
            cilindrada = cilindrada.replace("cc","").strip() if cilindrada != None else None
            potencia = potencia.replace("cv","").strip() if potencia != None else None
            placa_resource = cs.findAll("img")[0]['data-src']
            info =[venal,marca,modelo,importado,ano,cor,cilindrada,potencia,combustivel,uf,municipio,placa,placa_resource]
            print(json.dumps(info))
            return json.dumps(info)
        except Exception as error:
            return 'ocorreu um erro: '+str(error)
            
        else:
            abort(400)



@app.route('/ocr', methods=['POST'])
def ocr_placa():
    cropped = cv2.imread(diretorio+'/alpr_app/cropped.jpg')
    char=segment_characters(cropped)
    placa = show_results(char)
    print(json.dumps(placa))
    return json.dumps(placa) 

@app.route("/edit/<string:id>",methods=['GET'])
def edit(id):
    if request.method=='GET':
        con=sql.connect("banco_alpr.db")
        cur=con.cursor()
        agora = datetime.datetime.now()
        cur.execute("update historico_veiculos set HORA_SAIDA=? where ID=?",(agora,id))
        con.commit()
        flash('User Updated','success')
        return redirect(url_for("index"))
    

if __name__ == '__main__':
    app.secret_key='admin123'
    app.run()