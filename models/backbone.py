from models.simplified_efficientnetv2 import *
from models.resnet import *
from models.efficientnetv2 import *
import numpy as np

def efficientnetv2_s(num_classes,input_channel,stage):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    
    model_cnf_help = [[3, 3, 1, 1, 64, 64, 0, 0],
                    [2, 3, 1, 4, 64, 64, 0, 0],
                    ]
    model_config=np.array(model_cnf_help)
    model_config=model_config[stage].tolist()

    model = EfficientNetV2(model_cnf=model_config,
                           model_cnf_help=model_cnf_help,
                           num_classes=num_classes,
                           dropout_rate=0.2,input_channel=input_channel)
    return model

def efficientnetv2_sa(num_classes: int = 1000,input_channel=3):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    

    
    model_config = [[2, 3, 1, 1, 24, 24, 0, 0],
                    [4, 3, 2, 4, 24, 48, 0, 0],
                    [4, 3, 2, 4, 48, 64, 0, 0],
                    [6, 3, 2, 4, 64, 128, 1, 0.25],
                    [9, 3, 1, 6, 128, 160, 1, 0.25],
                    [15, 3, 2, 6, 160, 256, 1, 0.25]]

    model = EfficientNetV2a(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.2,input_channel=input_channel)
    return model
def resnet18(num_classes=1000,include_top=True,input_channel=3):
    return ResNet(BasicBlock,[2,2,2,2],num_classes=num_classes,include_top=include_top,input_channel=input_channel)

def resnet34(num_classes=1000, include_top=True,input_channel=3):
    
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top,input_channel=input_channel)

def resnet50(num_classes=1000, include_top=True,input_channel=3):
    
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top,input_channel=input_channel)