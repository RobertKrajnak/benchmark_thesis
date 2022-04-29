import os
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as Net

model = Net(weights='imagenet')

os.makedirs('./Desktop', exist_ok=True)

# Save the h5 file to path specified.
model.save("./Desktop/mobilenet.h5")
