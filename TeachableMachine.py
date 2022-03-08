import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2 as cv #Please install with PIP: pip install cv2

TM_DATA = None
model = None
cap = None
ret = None
frame = None
Prediction = None
key = None

print('START')
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
TM_DATA = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
cap = cv.VideoCapture(0)
while True:
  ret , frame = cap.read()
  cv.imshow('Window',frame)
  frame = cv.resize(frame, (224, 224))
  image_array = np.asarray(frame)
  # Normalize the image
  normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
  # Load the image into the array
  TM_DATA[0] = normalized_image_array
  Prediction = model.predict(TM_DATA)
  print('Planograma')
  print(round(Prediction[0][0]*100,2),"%")
  print('Ruffles de Queso')
  print(round(Prediction[0][1]*100,2),"%")
  print('Paquetaxo Quexo')
  print(round(Prediction[0][2]*100,2),"%")
  print('Fritos sal y limon')
  print(round(Prediction[0][3]*100,2),"%")
  print('Fritos chile y limon')
  print(round(Prediction[0][4]*100,2),"%")
  print('Cheetos Flaming Hot')
  print(round(Prediction[0][5]*100,2),"%")
  print('Sabritas originales')
  print(round(Prediction[0][6]*100,2),"%")
  print('Tostitos Flaming Hot')
  print(round(Prediction[0][7]*100,2),"%")
  print('Tostitos Salsa Verde')
  print(round(Prediction[0][8]*100,2),"%")
  print('Cheetos Azules')
  print(round(Prediction[0][9]*100,2),"%")
  print('Sabritas Adobadas')
  print(round(Prediction[0][10]*100,2),"%","\n")
  key = cv.waitKey(2000)
  if key == (ord('q')):
    break
cv.destroyAllWindows()
cap.release()
print('TNE END')

