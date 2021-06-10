from tflite_runtime.interpreter import Interpreter
from PIL import Image

import numpy as np
import picamera
import io
import time

interpreter = Interpreter(model_path="mask_classifier_2.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
floating_model = input_details[0]['dtype'] == np.float32
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

with picamera.PiCamera(resolution=(300, 300), framerate=30) as camera:
#     camera.rotation = 180
    camera.start_preview()
    try:
      stream = io.BytesIO()
      for _ in camera.capture_continuous(
          stream, format='jpeg', use_video_port=True):
        stream.seek(0)
        image = Image.open(stream).convert('RGB').resize((width, height))
        image=np.expand_dims(image, axis=0)
        input_data= (np.float32(image)-127.5)/127.5
        
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        details = interpreter.get_output_details()[0]
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)
        
        
        
        scalar_results = int(round(np.asscalar(results)))
        
        stream.truncate()
        print(details)
#         if scalar_results < 1:
#             print('No tiene')
# #             camera.annotate_text = 'No tiene tapa bocas'
# 
#         else:
#             print('Tiene')
# #             camera.annotate_text = 'Tiene tapa bocas'
        
    finally:
        camera.stop_preview()
        
        