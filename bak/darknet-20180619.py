from __future__ import division

import numpy as np
import re
from keras.layers import convolutional, LeakyReLU, BatchNormalization, Input
from keras.models import Model
import tensorflow as tf


# model = None            # The global variable for keras's model

def conv2d_with_bn(filters, kernel_size, stride, padding, activation, bias, batch_normalize):
    """
    As the guide of YOLOv3 implemented by PyTorch merge a convNet and a batch normalization
    as one layer, so I have to use this func to do this.
    :param filters: the dimensionality of the output space
    :param kernel_size: specifying the length of the 1D convolution window
    :param stride: specifying the stride length of the convolution
    :param padding: padding flag. Use padding or not.
    :param activation: choose the active function.
    :param bias:
    :param batch_normalize: BN flag. There is a BN layer following the convNet layer.

    :return: A function as a convNet layer followed by a BN layer.
    """

    # Check the activation
    if activation == "leaky":
        active = LeakyReLU(alpha=0.1)
    else:
        active = "linear"

    # Check use padding or not.
    if padding:
        pad = "same"
    else:
        pad = "valid"

    def layer_func(input):
        """
        Layer function. check the batch_normalize, if value is 1, then add a BN layer after convNet layer;
        otherwise, just return the output of conv layer.
        :param input: input tensor for layers
        :return: output tensor of conv layer or BN layer, depend on batch_normalize is 1 or 0.
        """
        conv_output = convolutional.Conv2D(filters=filters, kernel_size=kernel_size,
                                           strides=stride, padding=pad, activation=active, use_bias=bias)(input)
        if batch_normalize:
            # If batch_normalize flag rise, add a BN layer after the convNet layer,
            # or just return the convNet layer's output.
            bn_output = BatchNormalization()(conv_output)
            return bn_output
        else:
            return conv_output

    return layer_func


def bilinear_upsample(stride):
    """
    Use tf.image.resize_images function to implement bilinear upsampling of Pytorch.
    :param stride:  the multiplier for the image height / width
    :return: If images was 4-D, a 4-D float Tensor of shape [batch, new_height, new_width, channels].
             If images was 3-D, a 3-D float Tensor of shape [new_height, new_width, channels]
    """

    def upsample_layer_func(image):
        image_size = tf.shape(image)  # get the shape of tensor.
        new_image_size = np.int32(image_size[-3:-2] * stride)  # compute new size of tensor should be.
        new_image = tf.image.resize_images(image, new_image_size)  # use tf's function to bilinear upsampling.
        return new_image

    return upsample_layer_func


def parse_cfg(cfgfile):
    """
    Take a configuration file.

    Returns a list of blocks.Each blocks describes a block in the
    neural network to be built. 
    Block is represented as a dictionary in the list.

    cfgfile: a string of the path of cfg file.
    """

    # Open config file of network and read all lines.
    try:
        cfg_file = open(cfgfile, 'r')
        lines = cfg_file.readlines()
        cfg_file.close()
    except Exception:
        print("Can not open the cfgfile.")
        return

    block_list = []  # list for blocks
    net_dict = None  # as the block dict used for first loop

    for line in lines:
        # check for comments and empty line
        if line[0] == "#" or line == "\n":
            continue

        # get attributes from the readline.
        # As attributes part after block type, this part should follow
        # the block type part. But there are much more line of attributes than 
        # block type's, so I put it before the block type check wish could make
        # the code more efficient.
        match = re.search(r'(?P<key>\w+)(?: ?)=(?: ?)(?P<value>[\w.,\- ]+)', line)
        if match:
            # As the first line of config file is block type, this error would not be
            # touched when the first loop.
            # But it still there, still working on it.
            net_dict[match.group("key")] = match.group("value")
            # As this line describe the attribute of layer, 
            # no need to match the layer type again.
            continue

        # check for layer type.
        match = re.search(r'\[(?P<netType>\w+)\]', line)
        if match:
            if net_dict:
                # if it is not the first loop, put the dict in the 
                # end of block_list; and skip when it is first loop.
                block_list.append(net_dict)
            net_dict = {"type": match.group("netType")}
            # #As this line describe the type of layer,
            # # no need to match attribute in this again.
            # continue        
    return block_list


def load_weights(weightfile):
    """
    Load weight file for all layer.
    """
    # Open the weights file
    fp = open(weightfile, "rb")

    # The first 5 values are header information
    # 1. Major version number
    # 2. Minor version number
    # 3. subversion number
    # 4,5. Images seen by the network (during training)
    try:
        # Open the weights file
        fp = open(weightfile, "rb")

        header = np.fromfile(fp, dtype=np.int32, count=5)
        # self.header = torch.from_numpy(header)
        # self.seen = self.header[3]

        # Load all weights to array
        weights = np.fromfile(fp, dtype=np.float32)
    finally:
        fp.close()

    # TODO: load weight from file and initialization for model.


def create_model_list(blocks: list):
    """
    Create a list of model with blocks_list.
    Return a list, to make compute map.
    """

    # Captures the information about the input and pre-processing
    net_info = blocks[0]
    model_list = []
    prev_filters = 3
    # hold all output for shortcut and route
    # TODO: make this output filters smaller.
    output_filters = []

    # TODO: create the model list by blocks.
    for index, block in enumerate(blocks[1:]):
        if block["type"] == "convolutional":

            # Get info about the layer
            activation = block["activation"]

            try:
                batch_normalize = int(block["batch_normalize"])
                bias = False
            except Exception:
                batch_normalize = 0
                bias = True

            filters = int(block["filters"])
            padding = int(block["pad"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])

            layer = conv2d_with_bn(filters=filters, kernel_size=kernel_size, stride=stride, padding=padding,
                                   activation=activation, bias=bias, batch_normalize=batch_normalize)

        elif block["type"] == "upsample":
            # If it's a upsampling layer, we use Bilinear2dUpsampling.
            stride = int(block["stride"])
            filters = prev_filters  # keep the length of output_filter list same as model_list
            layer = bilinear_upsample(stride=stride)

        elif block["type"] == "route":
            # create route layer
            link_layers = block["layers"].split(',')
            start = int(link_layers[0])  # set Start of route

            # set end of route, if there exists one.
            try:
                end = int(link_layers[1])
            except Exception:
                end = 0

            # Positive annotation
            if start > 0:
                start = start - index

            if end > 0:
                end = end - index

            # TODO: create route layer
            layer = None

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif block["type"] == "shortcut":
            # create shortcut layer
            # TODO: create shortcut layer.
            layer = None
            filters = prev_filters  # shortcut layer has same shape of output tensor as previous layer.

        elif block["type"] == "yolo":
            # create detection layer, which is yolo layer in darknet framework.
            # get masks
            mask = block["mask"].split(',')
            mask = [int(x) for x in mask]

            # get anchors
            anchors = block["anchors"].split(',')
            # rebuild the anchor size as tuples
            anchors = np.int32([(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)])
            activing_anchors = [anchors[i] for i in mask]

            # TODO: create yolo layer.
            layer = None
            filters = None

        else:
            # miss match all layer types.
            # It shouldn't happen.
            continue

        model_list.append(layer)
        prev_filters = filters
        output_filters.append(filters)

    return net_info, model_list


def create_yolo_model(model_list: list, input_shape=(416, 416, 3)):
    """
    Create YOLOv3 model with list of layers which based on cfg_file.
    :param model_list: A list of layers which needed.
    :return: A functional Keras model
    """

    # Define the input as a tensor with shape input_shape
    X = Input(input_shape)

    # Catch the output of each layer used for route layer and shortcut layer.
    layer_output_list = []

    write = 0       # flag of getting the first output layer

    # Traverse the model_list
    for layer_index, layer in enumerate(model_list):
        # In model_list, both convNet layer and upsample layer are functional,
        # to let layer work, just need a input.
        # But route and shortcut layer, they need the output from other layers,
        # so, when create model_list, this two layers set as dict to hold info,
        # when building the model, get info from dict of layer, and get output
        # from layer_output_list to implement route or shortcut layer.
        layer_type = type(layer)    # get layer type
        if layer_type != dict:
            # If this layer is not route or shortcut layer, then
            X = layer(X)
        else:
            pass
        layer_output_list.append(X)     # append output in list


class Darknet(object):
    """
    Create Darknet of YOLOv3
    """

    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.module = None


def main():
    my_list = parse_cfg("./cfg/yolov3.cfg")
    _, model_list = create_model_list(my_list)
    print(model_list)


if __name__ == '__main__':
    main()
