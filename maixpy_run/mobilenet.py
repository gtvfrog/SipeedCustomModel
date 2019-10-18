#imports
import sensor, image, lcd, time
import KPU as kpu
#Iniciando sensores e tela
lcd.init()
sensor.reset()
#Setup camera/display
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((224, 224))
sensor.set_vflip(1)
#Limpando e desenhando na tela
lcd.clear()
lcd.draw_string(100,96,"Demonstraçao")
lcd.draw_string(100,112,"Carregando...")
#Abre o arquivo texto que contem os labels
f=open('labels.txt','r')
#Retorna todas as linhas do arquivo como lista
labels=f.readlines()
#Fecha o arquivo
f.close()
#Carrega o modelo
task = kpu.load("/sd/model.kmodel")

#Loop infinito
while(True):
    #Pega um frame da camera
    img = sensor.snapshot()
    #Frame processado no modelo gerando um map com os valores de inferencia
    fmap = kpu.forward(task, img)
    #transformar o map em lista
    plist=fmap[:]
    #pegar o maior valor de inferencia
    pmax=max(plist)
    #se o valor for maior que 75% mostra na tela
    if(pmax>0.75):
        #pega o item da lista com o maior valor de inferencia
        max_index=plist.index(pmax)
        #Desenha o nome e o valor de inferencia da tela
        a = img.draw_string(0,0, str(labels[max_index].strip()), color=(255,0,0), scale=2)
        a = img.draw_string(0,20, str(pmax), color=(255,0,0), scale=2)
        print((pmax, labels[max_index].strip()))
    #Mostra na tela
    a = lcd.display(img)
a = kpu.deinit(task)
