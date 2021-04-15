import torch

from icarl import iCaRLmodel
from parser import get_parser
from models.resnet import resnet18_cbam, resnet34_cbam


def main(args):
    convnet_types = {
        'resnet18': resnet18_cbam,
        'resnet34': resnet34_cbam
    }
    model_types = {
        'icarl': iCaRLmodel
    }
    class_settings = {
        '1': [50] + 50*[1],
        '2': [50] + 25*[2],
        '5': [50] + 10*[5],
        '10': [50] + 5*[10]
    }

    task_classes = class_settings[args['increment']]
    #task_classes = [10] * 10
    feature_extractor = convnet_types[args['convnet']] #TODO: Move this to iCaRL __init__
    img_size = 32
    model = iCaRLmodel(args, task_classes=task_classes, feature_extractor=feature_extractor)

    for i in range(len(task_classes)):
        print(f"Task {i}\n")
        model.beforeTrain()
        accuracy=model.train()
        model.afterTrain(accuracy)
    


if __name__ == '__main__':
    args = get_parser().parse_args()
    args = vars(args) 
    main(args)
