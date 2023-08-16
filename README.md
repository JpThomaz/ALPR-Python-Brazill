
# Sistema ALPR Para Estacionamentos

TCC "SISTEMA ALPR PARA ESTACIONAMENTOS" com foco em IA, Visão Computacional e Redes Neurais Convolucionais. Projeto que me ocasionou algumas noites mal dormidas mas também bem divertidas.
Praticamente o sistema ALPR é um sistema automático para reconhecimento de placas automotivas/veiculares e essa aplicação consiste em 5 etapas.
Captura do vídeo ou frame, reconhecimento da placa, segmentação de caracteres, reconhecimento de caracteres e o ultimo a validação da placa trazendo algumas informações do veículo.
O projeto foi feito todo em Python e utilizando algumas bibliotecas como OpenCV, Flask, BS4, Tensor Flow e KERAS. Para o reconhecimento de placa fiz um modelo de ML em YOLOv3, para segmentar os caracteres usei muita morfologia matemática e OpenCV e para Reconhecimento dos caracteres utilizei uma CNN (Rede Neural Convolucional) usando o KERAS. No final quando os caracteres da placa são reconhecidos, faço uma verificação em um crawler que criei que traz algumas informações publicas da placa como: valor da tabela fipe, modelo, cor, localização etc... Logo após essa verificação a entrada do veículo é dada no banco de dados, assim o único trabalho é dar baixa no veículo. 

## Funcionalidades

- Identificar Placa
- Segmentar Caracteres
- Reconhecer Caracteres
- Trazer innformações do Veículo (Crawler)
- Validar Placa
- Salvar informações no BD
- Entrada Automática de veículo

## Aplicação

![App Screenshot](https://github.com/JpThomaz/ALPR-Python-Brazill/blob/main/img/captura.png?raw=true)
![App Screenshot](https://github.com/JpThomaz/ALPR-Python-Brazill/blob/main/img/inserido.png?raw=true)
![App Screenshot](https://github.com/JpThomaz/ALPR-Python-Brazill/blob/main/img/saída.png?raw=true)

## Instalação

Recomendo que crie um ambiente virtual e instale as bibliotecas python

```bash
  pip install -r requirements.txt
```

Criar Banco de dados
```bash
  py create_db.py
```

Baixe abaixo o modelo já treinado da rede neural convolucional e salve em:
```bash
  \alpr_app\yolo_utils\lapi.weights
```
- [Baixar modelo](https://github.com/JpThomaz/ALPR-Python-Brazill/releases/download/model/lapi.weights)

O crawler roda no Firefox, por isso é necessario instalar o geckodriver. (Você também pode rodar no Chrome ou outro navegador)
Certifique-se que apóx baixar o driver, colocar o diretório dele como variável de ambiente.
- [Lib Geckodriver](https://github.com/mozilla/geckodriver/releases)

Após fazer todos os passos acima, execute  o server Flask
```bash
  py app.py
```
## Autor

- [@JpThomaz](https://github.com/JpThomaz)
- [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/joao-pedro-thomaz-de-paula/)


## Etiquetas

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)

