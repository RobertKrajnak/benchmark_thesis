import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image
import numpy as np
import glob
import time

# # ------------ load tf lite model ------------

# Specify the TensorFlow model, labels, and image
script_dir = pathlib.Path(__file__).parent.absolute()

model_file = os.path.join(script_dir, "../default_efficientdet_models_and_labels/efficientdet_lite3_model.tflite")
# model_file = os.path.join(script_dir, 'mobilenet_v2_1.0_224_quant.tflite')
# model_file = os.path.join(script_dir, 'mobilenet_v2_1.0_224_quant_edgetpu.tflite')

# label_file = os.path.join(script_dir, 'imagenet_labels.txt')
# image_file = os.path.join(script_dir, 'n01443537_11099_goldfish.jpg')

# Initialize the TF interpreter
interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()

# # --------------- Read the image preprocess and decode to a tensor -----------------

num_pic = [1, 2, 4, 8, 16, 32]
image_path = r'../../../data/Coco2017_val_samples/*.JPEG'

def get_infertime(int_numpic):
    iter = 0
    elapsed_time = []

    for filename in glob.iglob(image_path):
        if iter == num_pic[int_numpic]:
            break

        print("Predict picture number: ", iter + 1)
        print("Filename predict image: ", filename)

        # Resize the image
        size = common.input_size(interpreter)
        image = Image.open(filename).convert('RGB').resize(size, Image.ANTIALIAS)
        image = np.array(image, dtype="float32")
        # img = img / 255
        image /= 127.5
        image -= 1.

        # Run an inference
        common.set_input(interpreter, image)

        print("Start time")
        start_time = time.time()

        interpreter.invoke()

        end_time = time.time()
        print("Stop time")
        elapsed_time = np.append(elapsed_time, end_time - start_time)
        print("elapsed time for run is: ", elapsed_time)

        # # -------------------- Interpreting output tensor for our Image classification --------------

        classes = classify.get_classes(interpreter, top_k=3)

        # Print the result
        labels = dataset.read_label_file(label_file)
        for c in classes:
            print('%s: %.5f' % (labels.get(c.id, c.id), c.score))

        print(" ")
        iter += 1

    return elapsed_time.sum()

print("Run 2 Test iter - time is: ", get_infertime(1), " sec")
print("-----------------------------------------------------------")

for int_iter in range(len(num_pic)):
    print("Total time inference for " + str(num_pic[int_iter]) + " images is: ", get_infertime(int_iter), " sec")
    print("-----------------------------------------------------------")