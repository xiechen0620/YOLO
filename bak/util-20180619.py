from __future__ import division

import numpy as np
import tensorflow as tf


def predict_transform(prediction, inp_dim:int, anchors, num_classes):
    """
    prediction: a 'channel last' tensor, which means that shape is (B*W*H*C)
    prediction: our output
    inp_dim: input image height or width. get this parameter from 'net_info' - 'height'
    """
    predict_shape = prediction.shape
    batch_size = predict_shape[0]
    stride = inp_dim // predict_shape[1]
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    """this part should be double check with paper"""
    # change shape from (B*H*W*C) to (B*(H*W)*C)
    prediction_reshape = prediction.reshape(batch_size, grid_size*grid_size, bbox_attrs*num_anchors)
    # new_prediction = new_prediction.transpose(1, 2).contiguous()
    # change shape to (B*(H*W*num_anchors)*bbox_attrs)
    # bbox_attrs = 4+1+num_classes
    prediction_reshape = prediction_reshape.reshape(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    anchors_shape_list = [(anchor[0]/stride, anchor[1]/stride) for anchor in anchors]

    # sigmoid the centre_X, centre_Y, and object confidence
    # maybe should define a new tensor first.
    prediction_reshape[:, :, 0] = tf.sigmoid(prediction_reshape[:, :, 0])
    prediction_reshape[:, :, 1] = tf.sigmoid(prediction_reshape[:, :, 1])
    prediction_reshape[:, :, 4] = tf.sigmoid(prediction_reshape[:, :, 4])

    # Add the center offset
    # create the coordinates for all gride box
    grid = np.array(grid_size)
    x, y = np.meshgrid(grid, grid)

    # ture x and y coordinates to 1 columns.
    x_offset = x.reshape((-1, 1), order='C')
    y_offset = y.reshape((-1, 1), order='C')

    x_y_offset = np.concatenate((x_offset, y_offset), axis=1) #concatenate x,y array in one array[x,y]
    x_y_offset = x_y_offset.repeat(repeat=num_anchors, axis=1)  # repeat gride box coordinates for all anchor boxes
    x_y_offset = x_y_offset.reshape((-1, 2), order='C') # change shape to fit with preiction's shape.
    x_y_offset = np.expand_dims(x_y_offset, axis=0)     # insert a dimension before first axis

    # convert np.array to tensor
    x_y_offset_tensor = tf.convert_to_tensor(x_y_offset, dtype=tf.float32)
    
    # add offset of x and y on out_put
    prediction_reshape[:, :, :2] += x_y_offset_tensor
    
    
    """log sape transform height and width"""
    # change the list of tuple which is size of anchor box to tensor
    anchors_tensor = tf.convert_to_tensor(anchors_shape_list, dtype=tf.float32)
    
    # repeat anchors_tensor on axis=0 to fit output shape.
    # and insert a dimension before first axis
    anchors_tensor_output_shape = anchors_tensor.repeat(grid_size*grid_size, axis=0).expand_dims(axis=0)
    
    # compute the height and width of bounding box to get real size based on output image size.
    prediction_reshape[:, :, 2:4] = tf.exp(prediction_reshape[:, :, 2:4]) * anchors_tensor_output_shape

    # apply sigmoid activation to the class scores.
    prediction_reshape[:, :, 5: (5+num_classes)] = tf.sigmoid(prediction_reshape[:, :, 5:(5+num_classes)])

    # resize the detections map to the size of input image size.
    prediction_reshape[:, :, :4] *= stride

    # return transfored detection
    return prediction_reshape
