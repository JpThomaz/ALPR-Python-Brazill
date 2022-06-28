from paddleocr import PaddleOCR

def ocr(frame):
    try:
        ocr_model = PaddleOCR(lang='en')
        result = ocr_model.ocr(frame)
        result
        for res in result:
            print('placa ocr: ',res[1][0])
            res1=res[1][0]
        resultado= str(res1).replace('-','')
        resultado = resultado.replace('.','')
        resultado = resultado.replace(':','')
        return (resultado)
    except Exception as error:
        return ''





