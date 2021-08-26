from __future__ import print_function
from __future__ import division
import torch.nn as nn

import torch
from torchvision import models
import torch.nn.functional as Fun


class VariousModels:
    def __init__(self, model_name, num_classes, feature_extract):
        self.model_name = model_name
        self.num_classes = num_classes
        self.feature_extract = feature_extract

    def initialize_model(self, model_name, num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        if model_name == "resnet_multi_task":
            """ Resnet50
            """
            model_ft = models.resnet50(pretrained=use_pretrained)
            # model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # for variable image size
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features

            model_ft.fc = nn.Linear(num_ftrs, 512)
            input_size = 224

        elif model_name == "resnet":
            """ Resnet18
            """
            model_ft = models.resnet34(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 299

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size

    @staticmethod
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


class MultiOutputModel(nn.Module):
    def __init__(self, model_core, dd):
        super(MultiOutputModel, self).__init__()

        self.resnet_model = model_core

        self.x1 = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x1.weight)
        self.bn1 = nn.BatchNorm1d(256, eps=2e-1)

        self.x2 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x2.weight)
        self.bn2 = nn.BatchNorm1d(256, eps=2e-1)

        # heads
        self.y1o = nn.Linear(256, 3)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.y1o.weight)  #
        self.y2o = nn.Linear(256, 3)  # this is for number of font size class
        nn.init.xavier_normal_(self.y2o.weight)
        self.y3o = nn.Linear(256, 6)  # this is for number of font type class
        nn.init.xavier_normal_(self.y3o.weight)
        self.y4o = nn.Linear(256, 4)  # this is for number of font emphasis class
        nn.init.xavier_normal_(self.y4o.weight)

        self.d_out = nn.Dropout(dd)

    def forward(self, x):
        x1 = self.resnet_model(x)

        x1 = self.bn1(Fun.relu(self.x1(x1)))
        x1 = self.bn2(Fun.relu(self.x2(x1)))

        # heads
        y1o = Fun.softmax(self.y1o(x1), dim=1)
        y2o = Fun.softmax(self.y2o(x1), dim=1)
        y3o = Fun.softmax(self.y3o(x1), dim=1)
        y4o = Fun.softmax(self.y4o(x1), dim=1)  # should be sigmoid

        return y1o, y2o, y3o, y4o


class CombineMultiOutputModel(nn.Module):
    def __init__(self, model_core, dd):
        super(CombineMultiOutputModel, self).__init__()

        self.resnet_model = model_core

        self.x1 = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x1.weight)
        self.bn1 = nn.BatchNorm1d(256, eps=2e-1)

        self.x2 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x2.weight)
        self.bn2 = nn.BatchNorm1d(256, eps=2e-1)

        self.x11 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x11.weight)

        self.x12 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x12.weight)

        self.x13 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x13.weight)

        self.x14 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x14.weight)

        self.x21 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x21.weight)

        self.x22 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x22.weight)

        self.x23 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x23.weight)

        self.x24 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x24.weight)

        # heads
        self.y1o = nn.Linear(256, 3)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.y1o.weight)  #
        self.y2o = nn.Linear(256, 3)  # this is for number of font size class
        nn.init.xavier_normal_(self.y2o.weight)
        self.y3o = nn.Linear(256, 6)  # this is for number of font type class
        nn.init.xavier_normal_(self.y3o.weight)
        self.y4o = nn.Linear(256, 4)  # this is for number of font emphasis class
        nn.init.xavier_normal_(self.y4o.weight)

        self.d_out = nn.Dropout(dd)

    def forward(self, x1_input):
        x1_out = self.resnet_model(x1_input)
        #x1_out = self.bn1(Fun.relu(self.x1(x1_out)))
        #x1_out = self.bn2(Fun.relu(self.x2(x1_out)))

        #x2_out = self.resnet_model(x2_input)
        #x2_out = self.bn1(Fun.relu(self.x1(x2_out)))
        #x2_out = self.bn2(Fun.relu(self.x2(x2_out)))

        #x11 = self.bn1(Fun.relu(self.x11(x1_out)))
        #x12 = self.bn1(Fun.relu(self.x12(x1_out)))
        #x13 = self.bn1(Fun.relu(self.x13(x1_out)))
        #x14 = self.bn1(Fun.relu(self.x14(x1_out)))

        #x21 = self.bn1(Fun.relu(self.x21(x2_out)))
        #x22 = self.bn1(Fun.relu(self.x22(x2_out)))
        #x23 = self.bn1(Fun.relu(self.x23(x2_out)))
        #x24 = self.bn1(Fun.relu(self.x24(x2_out)))

        #x11 = x11 * x21
        #x12 = x12 * x22
        #x13 = x13 * x23
        #x14 = x14 * x24

        # heads
        #y1o = Fun.softmax(self.y1o(x11), dim=1)
        #y2o = Fun.softmax(self.y2o(x12), dim=1)
        #y3o = Fun.softmax(self.y3o(x13), dim=1)
        #y4o = Fun.softmax(self.y4o(x14), dim=1)  # should be sigmoid

        return x1_out


class SingleOutputModel(nn.Module):
    def __init__(self, model_core, dd, num_output_class):
        super(SingleOutputModel, self).__init__()

        self.resnet_model = model_core
        self.num_output_class = num_output_class  # to get the number of output classes

        self.x1 = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x1.weight)

        self.bn1 = nn.BatchNorm1d(256, eps=2e-1)
        self.x2 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x2.weight)
        self.bn2 = nn.BatchNorm1d(256, eps=2e-1)

        # heads
        self.y1o = nn.Linear(256, 3)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.y1o.weight, self.num_output_class)  #

        self.d_out = nn.Dropout(dd)

    def forward(self, x):
        x1 = self.resnet_model(x)

        x1 = self.bn1(Fun.relu(self.x1(x1)))
        x1 = self.bn2(Fun.relu(self.x2(x1)))

        # heads
        y1o = Fun.softmax(self.y1o(x1), dim=1)

        return y1o
