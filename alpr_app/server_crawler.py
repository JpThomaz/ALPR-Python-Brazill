from flask import Flask, request, abort,jsonify
import requests
import json
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from bs4 import BeautifulSoup as bs

app = Flask(__name__)
url = "https://placafipe.com/placa/"      
options = webdriver.FirefoxOptions()
options.add_argument('--headless')
driver = webdriver.Firefox(options=options)
executor_url  = driver.command_executor._url
session_id = driver.session_id
driver.get(url)
driver.implicitly_wait(30)
print('***** Rodando Crawler Placa *****')
@app.route('/', methods=['POST'])


def get_placa():
    
    if request.method == 'POST':
        print("received data: ", request.json)
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
            #driver.implicitly_wait(30)
            #options = webdriver.FirefoxOptions()
            #options.add_argument('--headless')
            #driver2 = webdriver.Remote(command_executor=executor_url,desired_capabilities={})#options=options)
            #driver2.session_id = session_id
            #driver = webdriver.Chrome(options=options) 
            driver.get(url + placa)
            #print(driver2.current_url)
            #driver2.current_url = url + placa
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
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
