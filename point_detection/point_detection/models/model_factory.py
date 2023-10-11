from torchvision.models import resnet, shufflenetv2, efficientnet, convnext, resnext50_32x4d, squeezenet
from torch import nn


def model_factory(hparams):
    """ Returns the given model

    Params:
        hparams (dict): Required parameters for the given model
    """
    pretrained = hparams['pretrained_model']
    out_features = 2
    model = None
    if hparams['model_type'].lower() == 'resnet18':
        if pretrained:
            model = resnet.resnet18(weights=resnet.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = resnet.resnet18(weights=None)
        model.fc = nn.Sequential(nn.Linear(in_features=512, out_features=out_features), nn.Sigmoid())  # type: ignore
    elif hparams['model_type'].lower() == 'shufflenet':
        if pretrained:
            model = shufflenetv2.shufflenet_v2_x1_0(weights=shufflenetv2.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        else:
            model = shufflenetv2.shufflenet_v2_x1_0(weights=None)
        model.fc = nn.Sequential(nn.Linear(in_features=1024, out_features=out_features), nn.Sigmoid())  # type: ignore
    elif hparams['model_type'].lower() == 'efficientnet':
        if pretrained:
            model = efficientnet.efficientnet_b0(weights=efficientnet.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            model = efficientnet.efficientnet_b0(weights=None)  # Assuming b0 variant
        model.classifier[1] = nn.Linear(in_features=1280, out_features=out_features)
        model.classifier.append(nn.Sigmoid())
    elif hparams['model_type'].lower() == 'convnexttiny':
        if pretrained:
            model = convnext.convnext_tiny(weights=convnext.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        else:
            model = convnext.convnext_tiny(weights=None)
        model.classifier[2] = nn.Sequential(nn.Linear(in_features=768, out_features=out_features))  # type: ignore
        model.classifier.append(nn.Sigmoid())
    elif hparams['model_type'].lower() == 'convnextlarge':
        if pretrained:
            model = convnext.convnext_large(weights=convnext.ConvNeXt_Large_Weights.IMAGENET1K_V1)
        else:
            model = convnext.convnext_large(weights=None)
        model.classifier[2] = nn.Sequential(nn.Linear(in_features=1536, out_features=out_features))  # type: ignore
        model.classifier.append(nn.Sigmoid())
    elif hparams['model_type'].lower() == 'resnext':
        if pretrained:
            model = resnet.resnext50_32x4d(weights=resnet.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        else:
            model = resnet.resnext50_32x4d(weights=None)
        model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=out_features), nn.Sigmoid())  # type: ignore
    else:
        raise NameError(f'Model {hparams["model_type"]} not supported')
    return model
