from tkinter import *
import tkinter.font

import argparse
import io
import time
import numpy as np
import picamera
import threading

from PIL import Image, ImageTk, ImageFont, ImageDraw
from tflite_runtime.interpreter import Interpreter

  
## GUI DEFINITIONS ##
global texto, letra
image = None
texto = ""
letra = ""
win = Tk()
camera = picamera.PiCamera(resolution=(640, 480), framerate=30)
camera.rotation = 180
win.title("Traductor de señas")
myFont = tkinter.font.Font(family = 'Helvetica', size = 12, weight = "bold")
myFont2 = tkinter.font.Font(family = 'Helvetica', size = 20, weight = "bold")
## Funciones ##
def load_labels(path = 'clasification/labels.txt'):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]


def camara():
    global letra
    
    stream = io.BytesIO()
    camera.capture(stream, format='jpeg', use_video_port=True)
    stream.seek(0)
    imagen = Image.open(stream).convert('RGB')
    image = Image.open(stream).convert('RGB').resize((width, height),
                                                         Image.ANTIALIAS)
    label_id, prob, tiempo=inferencia(image)
    draw = ImageDraw.Draw(imagen)
    font = ImageFont.truetype("FreeMono.ttf",35)
    draw.text((0,0), "Letra: "+labels[label_id], (255,255,255),font=font)
    draw.text((0,40), "Puntaje: "+str(prob), (255,255,255),font=font)
    draw.text((0,80), "Tiempo: "+str(tiempo)+"ms", (255,255,255),font=font)
    letra = labels[label_id]
    img = ImageTk.PhotoImage(image = imagen)
    lblVideo.configure(image = img)
    lblVideo.image = img
    
    lblVideo.after(2, camara)
    
    
def inferencia(image):
    start_time = time.time()
    results = classify_image(interpreter, image)
    elapsed_ms = (time.time() - start_time) * 1000
    label_id, prob = results[0]
    
    return label_id, prob, elapsed_ms
    
    
    

def inicio():
    confirmButton.configure(state = 'active')
    newoButton.configure(state = 'active')
    newpButton.configure(state = 'active')
    endButton.configure(state = 'active')
    inicioButton.configure(state = 'disabled')
    camara()

def confirma():
    global letra, texto
    
    texto = texto + letra
    
    lblTexto.configure(text = texto)
    
def newp():
    global letra, texto
    
    texto = texto + " " + letra
    
    lblTexto.configure(text = texto)

def newo():
    global letra, texto
    texto = ""
    texto = texto + letra
    
    lblTexto.configure(text = texto)
    
def end():
    win.destroy()
    camera.close()
## Inicializacion interprete ##

labels = load_labels()

interpreter = Interpreter('clasification/senas_classifier.tflite')
interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']    
    

## Disposicion ##

## Fila 1 ##
lblInfo1 = Label(win, text = "Traductor de señas", font = myFont)
lblInfo1.grid(column=0, row = 0,columnspan = 4)

## Fila 2 ##
inicioButton = Button(win, text = ' Iniciar traduccion', font = myFont, bg = 'bisque2', height = 1, width = 24, command = inicio)
inicioButton.grid(column=0, row = 1,columnspan = 4)

## Fila 3

lblVideo = Label(win)
lblVideo.grid(column = 0, row = 2,columnspan=4)

## Fila 4

confirmButton = Button(win, text = 'Confirmar', font = myFont, bg = 'bisque2', height = 1, width = 24,state = 'disable', command = confirma)
confirmButton.grid(column=0, row = 3)

newpButton = Button(win, text = ' Nueva Palabra', font = myFont, bg = 'bisque2', height = 1, width = 24,state = 'disable',command = newp)
newpButton.grid(column=1, row = 3)

newoButton = Button(win, text = ' Nueva Oracion', font = myFont, bg = 'bisque2', height = 1, width = 24,state = 'disable', command = newo)
newoButton.grid(column=3, row = 3)

# Fila 5

lblInfo = Label(win, text = "Oracion traducida:", font= myFont)
lblInfo.grid(column=0, row = 4, columnspan = 4)

## Fila 6

lblTexto = Label(win, text = "", font=myFont2)
lblTexto.grid(column=0, row = 5, columnspan = 4)

## Fila 7

endButton = Button(win, text = ' Finalizar', font = myFont, bg = 'bisque2', height = 1, width = 24,state = 'disable',command=end)
endButton.grid(column=0 ,row = 6,columnspan = 4)

win.mainloop()