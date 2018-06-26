from __future__ import division

import numpy as np
import tensorflow as tf


def predict_transform(prediction, inp_dim: int, anchors, num_classes):
    """
    Transfer tensor from convNet(B*H*W*C) to output shape(B*(H*W*num_anchors)*bounding_box_attributes) 
    which easier to read.
    The axis=1 of output shape's mean is all anchor box of all grid cell.
    It looks like reshape(H*W) to (H*W, 1) by C order, and each element replaced by anchors of this grid cell.
    The coordinate of centreX and centreY would transfer to image size axis.
    The size of bounding box would transfer to value based on image size.

    :param prediction: a 'channel last' tensor, which means that shape of our output from convNet is (B*H*W*C)
    :param inp_dim: input image height or width. get this parameter from 'net_info' - 'height'
    :param anchors: all anchors' size which used in this scale
    :param num_classes: number of classes which could be detected by this model.
    :return: transformed tensor of out_put
    """

    # prediction = tf.convert_to_tensor(prediction, dtype=tf.float32)

    predict_shape = prediction.get_shape().as_list()
    # batch_size = predict_shape[0]
    stride = inp_dim // predict_shape[1]
    grid_size = inp_dim // stride
    num_of_bounding_box_attributes = 5 + num_classes
    num_anchors = len(anchors)

    """this part should be double check with paper"""
    # change shape from (B*H*W*C) to (B*(H*W)*C)
    # new_prediction = prediction.reshape(batch_size, grid_size * grid_size,
    #                                         num_of_bounding_box_attributes * num_anchors)
    new_prediction = tf.reshape(prediction, [-1, grid_size * grid_size, num_of_bounding_box_attributes * num_anchors])
    # new_prediction = new_prediction.transpose(1, 2).contiguous()
    # change shape to (B*(H*W*num_anchors)*num_of_bounding_box_attributes)
    # num_of_bounding_box_attributes = 4+1+num_classes
    # new_prediction = new_prediction.reshape(batch_size, grid_size * grid_size * num_anchors,
    #                                             num_of_bounding_box_attributes)
    new_prediction = tf.reshape(new_prediction, [-1, grid_size * grid_size * num_anchors, num_of_bounding_box_attributes])

    # get all anchors' shape in a array.
    anchors_shape_array = np.array([(anchor[0] / stride, anchor[1] / stride) for anchor in anchors])

    # sigmoid the centre_X, centre_Y, and object confidence
    # maybe should define a new tensor first.
    # new_prediction[:, :, 0] = tf.sigmoid(new_prediction[:, :, 0])
    # new_prediction[:, :, 1] = tf.sigmoid(new_prediction[:, :, 1])
    # new_prediction[:, :, 4] = tf.sigmoid(new_prediction[:, :, 4])
    new_prediction[:, :, 0].assign(tf.sigmoid(new_prediction[:, :, 0]))
    new_prediction[:, :, 1].assign(tf.sigmoid(new_prediction[:, :, 1]))
    new_prediction[:, :, 4].assign(tf.sigmoid(new_prediction[:, :, 4]))

    # Add the center offset
    # create the coordinates for all gride box
    grid = np.array(grid_size)
    x, y = np.meshgrid(grid, grid)

    # turn x and y coordinates to 1 columns.
    x_offset = x.reshape((-1, 1), order='C')
    y_offset = y.reshape((-1, 1), order='C')

    x_y_offset = np.concatenate((x_offset, y_offset), axis=1)  # concatenate x,y array in one array[x,y]
    x_y_offset = x_y_offset.repeat(repeats=num_anchors, axis=1)  # repeat gride box coordinates for all anchor boxes
    x_y_offset = x_y_offset.reshape((-1, 2), order='C')  # change shape to fit with preiction's shape.
    x_y_offset = np.expand_dims(x_y_offset, axis=0)  # insert a dimension before first axis

    # convert np.array to tensor
    x_y_offset_tensor = tf.convert_to_tensor(x_y_offset, dtype=tf.float32)

    # add offset of x and y on out_put
    new_prediction[:, :, :2] += x_y_offset_tensor

    """log sape transform height and width"""
    # repeat anchors_tensor on axis=0 to fit with output shape.
    anchors_shape_array = anchors_shape_array.repeat(repeats=grid_size * grid_size, axis=0)
    anchors_shape_array = np.expand_dims(anchors_shape_array, axis=0)       # insert a dimension before first axis.

    # change the list of tuple which is size of anchor box to tensor
    anchors_tensor = tf.convert_to_tensor(anchors_shape_array, dtype=tf.float32)

    # compute the height and width of bounding box to get real size based on output image size.
    new_prediction[:, :, 2:4] = tf.exp(new_prediction[:, :, 2:4]) * anchors_tensor

    # apply sigmoid activation to the class scores.
    new_prediction[:, :, 5: (5 + num_classes)] = tf.sigmoid(new_prediction[:, :, 5:(5 + num_classes)])

    # resize the detections map to the size of input image size.
    # TODO: tf.scatter_mul 
    new_prediction[:, :, :4] *= stride

    # return transfored detection
    return new_prediction
