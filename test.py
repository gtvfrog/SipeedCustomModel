#Importações
import keras
import numpy as np
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
#Modelo MobileNet, com os pesos pré-treinados no ImageNet
mobile = keras.applications.mobilenet.MobileNet()

#Função que prepara a imagem para o modelo
def prepare_image(file):
    img_path = ''
    #Carrega a imagem com o tamanho do modelo de 224,244
    img = image.load_img(img_path + file, target_size=(224, 224))
    #Converte a imagem em matriz
    img_array = image.img_to_array(img)
    #Salva a imagem
    image.save_img(img_path + file, img_array)
    #Expande a forma da matriz para utilizar na função do keras
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    #Adequa a imagem ao formato requerido pelo modelo
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

#Chama a função que prepara a imagem
preprocessed_image = prepare_image('PastorAlemao.jpg')
#Gera predições de saída para a imagem
predictions = mobile.predict(preprocessed_image)
#Decodifica as predições
results = imagenet_utils.decode_predictions(predictions)
#Mostra o resultado das predições
print(results)

preprocessed_image = prepare_image('PapaiNoel.jpg')
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
print(results)

preprocessed_image = prepare_image('Uno.jpg')
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
print(results)
