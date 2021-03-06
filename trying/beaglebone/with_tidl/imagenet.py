""" Process each frame using a single ExecutionObject.
    Increase throughput by using multiple ExecutionObjects.
"""

import os
import argparse
import json
import heapq
import logging
import numpy as np
import cv2

from tidl import DeviceId, DeviceType, Configuration, TidlError
from tidl import Executor, ExecutionObjectPipeline
from tidl import allocate_memory, free_memory


def main():
    """Read the configuration and run the network"""
    #logging.basicConfig(level=logging.INFO)

    args = parse_args()

    # config_file = 'tidl_config_j11_v2.txt'
    config_file = 'tidl_config_mobileNet2.txt'
    # config_file = 'tidl_config_mobileNet1.txt'
    labels_file = 'imagenet_objects.json'

    configuration = Configuration()
    status = configuration.read_from_file(config_file)
    
    print("Network needs",configuration.network_heap_size/1024**2,"+",configuration.param_heap_size/1024**2,"mb heap")
    

    if os.path.isfile(args.input_file):
        configuration.in_data = args.input_file
    else:
        print('Input image {} does not exist'.format(args.input_file))
        return
    print('Input: {}'.format(args.input_file))

    num_eve = Executor.get_num_devices(DeviceType.EVE)
    num_dsp = Executor.get_num_devices(DeviceType.DSP)

    if num_eve == 0 and num_dsp == 0:
        print('No TIDL API capable devices available')
        return

    # use 1 EVE or DSP since input is a single image
    # If input is a stream of images, feel free to use all EVEs and/or DSPs
    if num_eve > 0:
        num_eve = 1
        num_dsp = 0
    else:
        num_dsp = 1

    run(num_eve, num_dsp, configuration, labels_file)

    return


DESCRIPTION = 'Run the imagenet network on input image.'
DEFAULT_INFILE = 'n01443537_11099_goldfish.jpg'
# DEFAULT_INFILE = 'cat-pet-animal-domestic-104827.jpeg'

def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('-i', '--input_file',
                        default=DEFAULT_INFILE,
                        help='input image file (that OpenCV can read)')
    args = parser.parse_args()

    return args

PIPELINE_DEPTH = 2

def run(num_eve, num_dsp, configuration, labels_file):
    """ Run the network on the specified device type and number of devices"""

    print('Running network across {} EVEs, {} DSPs'.format(num_eve,
                                                                  num_dsp))

    dsp_device_ids = set([DeviceId.ID0, DeviceId.ID1,
                          DeviceId.ID2, DeviceId.ID3][0:num_dsp])
    eve_device_ids = set([DeviceId.ID0, DeviceId.ID1,
                          DeviceId.ID2, DeviceId.ID3][0:num_eve])

    # Heap sizes for this network determined using Configuration.showHeapStats
    configuration.param_heap_size = (3 << 23) #  # 0x180000000
    configuration.network_heap_size = (20 << 23)


    try:
        print('TIDL API: performing one time initialization ...')

        # Collect all EOs from EVE and DSP executors
        eos = []

        if eve_device_ids:
            eve = Executor(DeviceType.EVE, eve_device_ids, configuration, 1)
            for i in range(eve.get_num_execution_objects()):
                eos.append(eve.at(i))

        if dsp_device_ids:
            dsp = Executor(DeviceType.DSP, dsp_device_ids, configuration, 1)
            for i in range(dsp.get_num_execution_objects()):
                eos.append(dsp.at(i))

        eops = []
        num_eos = len(eos)
        for j in range(PIPELINE_DEPTH):
            for i in range(num_eos):
                eops.append(ExecutionObjectPipeline([eos[i]]))

        allocate_memory(eops)

        # open labels file
        with open(labels_file) as json_file:
            labels_data = json.load(json_file)

        configuration.num_frames = 1
        print('TIDL API: processing {} input frames ...'.format(
                                                     configuration.num_frames))

        num_eops = len(eops)
        for frame_index in range(configuration.num_frames+num_eops):
            eop = eops[frame_index % num_eops]

            if eop.process_frame_wait():
                process_output(eop, labels_data)

            if read_frame(eop, frame_index, configuration):
                eop.process_frame_start_async()

        free_memory(eops)

    except TidlError as err:
        print(err)

def read_frame(eo, frame_index, configuration):
    """Read a frame into the ExecutionObject input buffer"""

    if frame_index >= configuration.num_frames:
        return False

    # Read into the EO's input buffer
    arg_info = eo.get_input_buffer()
    np_arg = np.asarray(arg_info)

    img = cv2.imread(configuration.in_data)
    resized = cv2.resize(img, (224, 224))
    
    # resized = np.array(resized, dtype='float32')
    # resized = resized/127.5
    # resized = resized - 1.
    
    print(resized.shape)
    print(np.max(resized))
    print(np.min(resized))
    
    b_frame, g_frame, r_frame = cv2.split(resized)
    np_arg[0*224*224:1*224*224] = np.reshape(b_frame, 224*224)
    np_arg[1*224*224:2*224*224] = np.reshape(g_frame, 224*224)
    np_arg[2*224*224:3*224*224] = np.reshape(r_frame, 224*224)
    
    print(np_arg)
    print(np_arg.shape)

    eo.set_frame_index(frame_index)

    return True

def process_output(eo, labels_data):
    """Display the inference result using labels."""

    # keep top k predictions in heap
    k = 5
    # output predictions with probability of 10/255 or higher
    threshold = 10

    out_buffer = eo.get_output_buffer()
    output_array = np.asarray(out_buffer)

    k_heap = []
    for i in range(k):
        heapq.heappush(k_heap, (output_array[i], i))

    for i in range(k, out_buffer.size()):
        if output_array[i] > k_heap[0][0]:
            heapq.heappushpop(k_heap, (output_array[i], i))

    k_sorted = []
    for i in range(k):
        k_sorted.insert(0, heapq.heappop(k_heap))
        
    print("k_sorted: ", k_sorted)

    for i in range(k):
        if k_sorted[i][0] > threshold:
            print('{}: {},   prob = {:5.2f}%'.format(i+1, \
                             labels_data['objects'][k_sorted[i][1]]['label'], \
                             k_sorted[i][0]/255.0*100))

    return 0

if __name__ == '__main__':
    main()