from __future__ import division

import numpy as np
import re
import itertools


model = None            # The global variable for keras's model

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
    except:
        pass
    finally:
        cfg_file.close()

    block_list = []      # list for blocks
    net_dict = None     # as the block dict used for first loop

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
            net_dict = {}
            net_dict["type"] = match.group("netType")
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
    except:
        pass
    finally:
        fp.close()
    
    # TODO: load weight from file and initialization for model.


def create_model_list(blocks:list):
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
        if (block["type"]=="convolutional"):
            # TODO: create convolutional layer
        elif (block["type"]=="upsample"):
            # TODO: create upsample layer.
        elif (block["type"]=="route"):
            link_layers = block["layers"].split(',')
            start = int(link_layers[0])     # set Start of route

            # set end of route, if there exists one.
            try:
                end = int(link_layers[1])
            except:
                end = 0
            
            # Positive anotaion
            if start > 0:
                start = start - index
            
            if end > 0:
                end = end - index
            
            # TODO: create route layer

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
            
            # filters = end<0 ? output_filters[index+start]+output_filters[index+end] : output_filters[index+start]

        elif (block["type"]=="shortcut"):
            # TODO: create shortcut layer.
        elif (block["type"]=="yolo"):
            # TODO: create yolo layer.
        else:
            pass


    
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


if __name__ == '__main__':
    main()