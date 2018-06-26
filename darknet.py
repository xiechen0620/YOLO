from __future__ import division

import re

import numpy as np
from keras.layers import convolutional, LeakyReLU, BatchNormalization, Input, Concatenate, Lambda, Add, Activation
from keras.models import Model
from keras.utils import plot_model
import tensorflow as tf
import cv2

from util import predict_transform


def conv2d_with_bn(filters, kernel_size, stride, padding, activation, bias, batch_normalize, layer_index):
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
    :param layer_index: order of this block.

    :return: A function as a convNet layer with BN layer depends on flag.
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
        otherwise, just conv layer. Then add an Activation layer and return it's output.
        
        :param input: input tensor for layers
        :return: output tensor of Activation layer.
        """
        conv_layer = convolutional.Conv2D(filters=filters, kernel_size=kernel_size,
                                          strides=stride, padding=pad, use_bias=bias, name="conv2d_" + str(layer_index))
        mid_output = conv_layer(input)

        if batch_normalize:
            # If batch_normalize flag rise, add a BN layer after the convNet layer,
            # or just return the convNet layer's output.
            mid_output = BatchNormalization(name="batch_normalization_" + str(layer_index))(mid_output)

        # add Activation to conv or BN layer.
        activation_output = Activation(activation=active, name="activation_" + str(layer_index))(mid_output)

        return activation_output

    return layer_func


def bilinear_upsample(stride: int):
    """
    Use tf.image.resize_images function to implement bilinear upsampling of Pytorch.
    
    :param stride:  the multiplier for the image height / width
    :return: an evaluate fucntion for Lambda layer. take a image tensor as input and 
             will return a upsampled image tensor created by tf.image.resize_image().
    """

    def upsample_layer_func(image):
        """
        Layer function. Get the height and width from image's shape(B*H*W*C).
        Then, compute the new image's height and width by stride.
        Last, use TensorFlow's resize_image function to get new image by bilinear upsample
        
        :param image: Matrix tensor. It's a channel last image matrix computed by ConvNet.
        :return: If images was 4-D, a 4-D float Tensor of shape [batch, new_height, new_width, channels].
                 If images was 3-D, a 3-D float Tensor of shape [new_height, new_width, channels]
        """
        image_size = np.array(image.get_shape().as_list())          # get the shape of tensor as array.
        new_image_size = np.int32(image_size[-3:-1] * stride)       # compute new size of tensor should be.
        new_image = tf.image.resize_images(image, new_image_size)   # use tf's function to bilinear upsampling.
        return new_image

    return upsample_layer_func


def bilinear_upsample_shape(stride: int):
    """
    As the bilinear_upsample_layer built by Lambda layer and default output shape of Lambda
    layer is the input shape. So I have use a output_shape_function to return new shape of 
    the image upsampled by this layer for Keras create map.
    Upsample layer has not parameter to learn.
    
    :param stride: Expansion multiple of height and width.
    :return: a function to get output's shape.
    """

    def upsample_layer_shape_func(input_shape):
        """
        Compute output shape function.
        Take input's shape from Lambda layer, and retuen the output's shape for creating map.
        
        :param input_shape: the input's shape of Lambda layer.
        :return: a tuple. new shape of upsampled image. (B*new_H*new_W*C)
        """
        shape = np.array(input_shape)   # turn input_shape to array
        shape[-3:-1] *= stride          # comput new_H and new_W of new image
        return tuple(shape)

    return upsample_layer_shape_func


def parse_cfg(cfgfile):
    """
    Take a configuration file. read all layers' infor and re-storage to a list of dicts, 
    each dict is a layer. 
    All attribute of layer in config file will described as key-value in layer's dict.
    Order of dicts in list is same as layers' order in config file.

    :param cfgfile: a string of the path of cfg file.
    :return: A list of layers. 
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
    net_dict = None  # block dict used for first loop

    # Traverse lines, get all layers and its attributes in one dict(net_dict) as a block.
    # then append net_dict into block_list if get another layer.
    for line in lines:

        if line[0] == "#" or line == "\n":
            # check for comments and empty line
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
    else:
        # When finish traverse of lines, the lastest layer dict doesn't append into
        # block_list. So do it at here.
        block_list.append(net_dict)

    return block_list


def load_weights(weight_file: str, model: Model, conv_index: int):
    """
    Load weight from weight_file into model.
    The conv and BN layer weight's shape which getting by layer's name, as size of data getting from weight_file.
    Then, reset layer weight of conv and BN layer.

    :param conv_index:
    :param model:
    :param weight_file:
    :return:
    """

    # Open the weights file
    fp = open(weight_file, "rb")

    # The first 5 values are header information
    # 1. Major version number
    # 2. Minor version number
    # 3. subversion number
    # 4,5. Images seen by the network (during training)

    header = np.fromfile(fp, dtype=np.int32, count=5)  # read first 5 values
    header = tf.convert_to_tensor(header)
    seen = header[3]

    # Load all weights to array
    # weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    for index in range(conv_index):
        conv_layer = model.get_layer(name="conv2d_" + str(index + 1))  # get conv layer by order.
        conv_weights = conv_layer.get_weights()  # get layer's weights

    #
        try:
            bn_layer = model.get_layer(name="batch_normalization_" + str(index + 1))
            with_bn = True
            bn_weights = bn_layer.get_weights()
        except Exception:
            with_bn = False

        if with_bn:
            print("loading weight to: " + bn_layer.name)
            bn_weight_size = bn_weights[0].size
            bn_weight_shape = bn_weights[0].shape

            bn_bias = np.fromfile(fp, dtype=np.float32, count=bn_weight_size)
            bn_bias = np.reshape(bn_bias, newshape=bn_weight_shape)

            bn_weight = np.fromfile(fp, dtype=np.float32, count=bn_weight_size)
            bn_weight = np.reshape(bn_weight, newshape=bn_weight_shape)

            bn_running_mean = np.fromfile(fp, dtype=np.float32, count=bn_weight_size)
            bn_running_mean = np.reshape(bn_running_mean, newshape=bn_weight_shape)

            bn_running_var = np.fromfile(fp, dtype=np.float32, count=bn_weight_size)
            bn_running_var = np.reshape(bn_running_var, newshape=bn_weight_shape)

            # batch normalization's weight order is [gamma, beta, mean, std]
            # gamma is equal to weight, and beta equal to bias.
            # https://stackoverflow.com/questions/42793792/how-to-set-weights-of-the-batch-normalization-layer
            bn_weights = [bn_weight, bn_bias, bn_running_mean, bn_running_var]

            print("loading weight to: " + conv_layer.name)
            conv_weight_size = conv_weights[0].size
            conv_weight_shape = conv_weights[0].shape

            conv_weights[0] = np.fromfile(fp, dtype=np.float32, count=conv_weight_size)
            conv_weights[0] = np.reshape(conv_weights[0], newshape=conv_weight_shape)

            conv_layer.set_weights(conv_weights)
            bn_layer.set_weights(bn_weights)

        else:
            conv_weight = conv_weights[0]
            conv_bias = conv_weights[1]

            print("loading weight to: " + conv_layer.name)
            conv_weight = np.fromfile(fp, dtype=np.float32, count=conv_weight.size)
            conv_weight = np.reshape(conv_weight, newshape=conv_weights[0].shape)

            conv_bias = np.fromfile(fp, dtype=np.float32, count=conv_bias.size)
            conv_bias = np.reshape(conv_bias, newshape=conv_weights[1].shape)

            conv_weights = [conv_weight, conv_bias]

            conv_layer.set_weights(conv_weights)
    else:
        print("Loading weight file succeed")

    fp.close()


def create_model_list(blocks: list):
    """
    As create model by block_list directly would leave may variables which don't need for model,
    I define this function to build a list of layer.

    Here, I build the model using Function API instead of Sequential model for these reason:
          1. In offical code of YOLO v3 of Darknet, a Conv layer followed by a BN layer is treated
             as a single Conv layer with BN, and a Conv layer only is treated as a single Conv layer
             without BN. But in Keras, Conv layer is one layer and BN is other layer.
             If using Sequential Model, it is difficult to keep layer number of Keras Model
             consistent with offical model.
             It's easy to get wrong in route layer and shortcut layer.

             So, I use Function API to keep order of blocks in Keras consistent with offical model.
             And it seems like easy to build route layer and shortcut layer.

          2. For now, the output of model has three output for three scales. May be I could put them
             together in one main_output by FC layer or something else, but compute map will has
             three route for three scales.

             so, I had to use Function API, unless put all three scales part in one Function model
             as a layer of Sequential Model. I'll try this later.
         
    In layer list, Conv layer and upsample layer are functional layers which could put in model directly with a input;
    route layer and shortcut layer need other layers' output as input, so they have to be created when building model;
    yolo layer is unfinished yet.
    So route layer, shortcut layer and yolo layer are dict in block_list, the functional layer
    would be created when building the model.

    :param blocks: layer list. The layer info from cfg_file.
    :return: a dict of net_info, and a list of layer_func a list of layer_func(convolution layer and upsampling layer) and 
    dict(route, shortcut and yolo layers). The model will be built according to this list.
    """

    # Captures the information about the input and pre-processing
    net_info = blocks[0]
    model_list = []  # list of layer_func
    prev_filters = 3  # as input is image array, here set number of channels(RGB) as initialization.
    conv_index = 0  # count the number of conv layer

    # TODO: make this output filters smaller.
    output_filters = []  # hold all output for shortcut and route

    # As first element of block_list(blocks[0]) stores net_info, traverse block_list[1:], 
    # bulid layer func or layer dict for each layer.
    # then append it into model_list. 
    for index, block in enumerate(blocks[1:]):
        if block["type"] == "convolutional":
            conv_index += 1  # count number of this conv layer
            # It's a conv layer
            # Get attributes of layer
            activation = block["activation"]

            # check this conv layer with or without BN layer.
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

            # create a layer_func with a funciton.
            layer = conv2d_with_bn(filters=filters, kernel_size=kernel_size, stride=stride, padding=padding,
                                   activation=activation, bias=bias, batch_normalize=batch_normalize,
                                   layer_index=conv_index)

        elif block["type"] == "upsample":
            # If it's a upsampling layer, we use Bilinear2dUpsampling.
            stride = int(block["stride"])
            filters = prev_filters  # keep the length of output_filter list same as model_list

            # use tensorflow function create Lambda layer.  
            layer = Lambda(bilinear_upsample(stride=stride), output_shape=bilinear_upsample_shape(stride=stride),
                           name="bilinear_upsample_" + str(index))

        elif block["type"] == "route":
            # create route layer
            link_layers = block["layers"].split(',')
            start = int(link_layers[0])  # set Start of route

            # set end of route, if there exists one.
            try:
                end = int(link_layers[1])
            except Exception:
                end = 0

            # Positive annotation. 
            # Change positive layer number to negative value relativing to this layer.
            if start > 0:
                start = start - index

            if end > 0:
                end = end - index

            # Put necessary info in dict as route layer
            # Note: start layer and end layer(if exist) are using negative value.
            layer = {"layerType": "route", "start": start}

            # set output filters of route layer
            # if has end layer, set it in dict and output filters is the sum of two layers.
            # otherwise, just set filters of start layer as this layer's output layers.
            if end < 0:
                layer["end"] = end
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif block["type"] == "shortcut":
            # create shortcut layer
            layer = {"layerType": "shortcut", "from": int(block["from"]), "activation": block["activation"]}

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
            layer = {"layerType": "yolo", "anchors": activing_anchors, "classes": int(block["classes"])}

            # Don't matter what value should be, next layer should be route to get output from previous layer
            filters = prev_filters

        else:
            # miss match all layer types.
            # It shouldn't happen.
            continue

        model_list.append(layer)
        prev_filters = filters
        output_filters.append(filters)

    return net_info, model_list, conv_index


def create_yolo_model(net_info: dict, model_list: list, input_shape=(416, 416, 3)):
    """
    Create YOLOv3 model with list of layers which based on cfg_file.

    :param net_info:
    :param input_shape: shape of input image.
    :param model_list: A list of layers which needed.
    :return: A functional Keras model
    """

    # Define the input as a tensor with shape input_shape
    # input data shape is: (?, 416, 416, 3)
    input_X = Input(input_shape)
    X = input_X  # temporary variable. Used fro recursive.

    # Temporary variable. Get all three detection layers' output.
    detections = []

    # Catch the output of each layer used for route layer and shortcut layer.
    layer_output_list = []

    write = False  # flag of getting the first output layer

    # Traverse the model_list
    for layer_index, layer in enumerate(model_list):
        # In model_list, both convNet layer and upsample layer are functional,
        # to let layer work, just need a input.
        # But route and shortcut layer, they need the output from other layers,
        # so, when create model_list, this two layers set as dict to hold info,
        # when building the model, get info from dict of layer, and get output
        # from layer_output_list to implement route or shortcut layer.
        layer_type = type(layer)  # get layer type
        if layer_type != dict:
            # This layer is  convolution or upsampling layer.
            X = layer(X)  # just get layer a input and get the output

        else:
            # This layer is route or shortcut or yolo layer.
            if layer["layerType"] == "route":  # This layer is route layer.
                route_start = layer["start"]  # get start layer index. Negative value
                map_start = layer_output_list[layer_index + route_start]  # get output tensor from start layer.

                try:
                    # try to get end layer index. If exist, get the index and output tensor of that layer.
                    route_end = layer["end"]  # Negative value
                    map_end = layer_output_list[layer_index + route_end]

                    assert map_start.get_shape().as_list()[:-1] == map_end.get_shape().as_list()[:-1]

                    # out put of route layer is concatenate two layers' output tensor at channel axis.
                    X = Concatenate(axis=-1)([map_start, map_end])
                except Exception:
                    # If there is only start layer, the output tensor is start layer's output.
                    X = map_start

            elif layer["layerType"] == "shortcut":  # This layer is shortcut layer.
                shortcut_from = layer["from"]  # get add layer's index

                layer_a = layer_output_list[layer_index - 1]
                layer_b = layer_output_list[layer_index + shortcut_from]

                assert layer_a.get_shape().as_list() == layer_b.get_shape().as_list()

                # the sum of from_layer and previous layer as shortcut layer's output.
                X = Add()([layer_a, layer_b])

            elif layer["layerType"] == "yolo":  # yolo layer
                anchors: list = layer["anchors"]  # get anchors' size

                inp_dim = int(net_info["height"])  # Get input image dimension
                num_classes = layer["classes"]  # Get the number of classes

                # Transfer convNet output format to readable format
                # X = predict_transform(X, inp_dim=inp_dim, anchors=anchors, num_classes=num_classes)

                # check write flag which marks the model get first detection tensor or not.
                # if not write:
                #     # if model hasn't get first detection, create detection tensor and
                #     # raise the write flag.
                #     detections = X
                #     write = True
                # else:
                #     # if model has got detection, concatenate detection tensor and
                #     # new X tensor at axis=1
                #     detections = tf.concat([detections, X], axis=1)

                detections.append(X)
            else:
                # error
                continue

        layer_output_list.append(X)  # append output in list

    model = Model(inputs=input_X, outputs=detections, name='YOLO')

    return model


def get_test_image():
    img = cv2.imread("./img/test.png")
    img = cv2.resize(img, (416, 416))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    # img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

    return img


def main():
    my_list = parse_cfg("./cfg/yolov3.cfg")
    net_info, model_list, conv_index = create_model_list(my_list)
    # print(my_list)

    yolo_model = create_yolo_model(net_info, model_list)
    load_weights(weight_file="./weights/yolov3.weights", model=yolo_model, conv_index=conv_index)

    # for layer in yolo_model.layers:
    #     # print(type(layer).__name__)     # get layer's type
    #     print(layer.name)
    #     weight = layer.get_weights()
    #     print(type(weight))

    # plot_model(yolo_model, show_shapes=True)

    yolo_model.compile(optimizer="Adam", loss="binary_crossentropy")
    yolo_model.summary()
    #
    # for layer_index in range(conv_index):
    #     print(yolo_model.layers[layer_index+1].name)
    #     weights = yolo_model.layers[layer_index+1].get_weights()
    #     for weight in weights:
    #         print(weight.shape)


    # for layer in yolo_model.layers:
    #     print(layer.name)
    #     weights = layer.get_weights()
    #     for weight in weights:
    #         print(weight.shape)

    test_img = get_test_image()
    test = yolo_model.predict(test_img)
    #
    print(type(test))
    print(len(test))
    for element in test:
        print(element.shape)


if __name__ == '__main__':
    main()
