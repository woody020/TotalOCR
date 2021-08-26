from collections import OrderedDict

import torch
import logging
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from trainer import Trainer
from torchvision import transforms
#from word_image_datasets import WordImageDS
#from word_image_dataset_small import WordImageDS as wordSmallDataset
from multi_image_data_loader import WordImageDS as multiloaderDataset
from network_model import VariousModels, MultiOutputModel, CombineMultiOutputModel
#from logger import Logger
import os
import os.path
from os import path
import shutil


def load_state_dict(model_dir, is_multi_gpu):
    state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)['state_dict']
    if is_multi_gpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

def create_save_multi_dataset_together():
    # This following part is for generating the data by using the dataloader and to save it
    training_imag_paths = "dataset/train"
    validation_imag_paths_word = "dataset/test"
    #validation_imag_paths_patch = "dataset/test/"

    #validation_imag_paths_word_extra = "/home/mondal/Videos/Dataset/L3i_Text_Copies/Validation_Data_Word_Extra/"
    #validation_imag_paths_patch_extra = "/home/mondal/Videos/Dataset/L3i_Text_Copies/Validation_Data_Patch_Extra/"

    if path.isdir(training_imag_paths):
        print("Training path exists")
    else:
        raise SystemExit('Training path doesnt exists')

    if path.isdir(validation_imag_paths_word):
        print("Validation path exists")
    else:
        raise Exception('Validation path doesnt exists')

    data_transforms = {

        'train': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_imgs_obj = multiloaderDataset(training_imag_paths, transform=data_transforms['train'])

    validation_imgs_obj = multiloaderDataset(validation_imag_paths_word, transform=data_transforms['val'])


    mn_dataset_loader_train = torch.utils.data.DataLoader(dataset=train_imgs_obj,
                                                          batch_size=200, shuffle=True, num_workers=5)
    mn_dataset_loader_validation = torch.utils.data.DataLoader(dataset=validation_imgs_obj,
                                                               batch_size=100, shuffle=True, num_workers=5)
    print("The size of train dataset:", len(train_imgs_obj))
    print("The size of train dataset:", len(validation_imgs_obj))
    print(train_imgs_obj[0])

    return mn_dataset_loader_train, mn_dataset_loader_validation

def main():
    #logger = Logger('./logs/'+"save_training_params_crossnet_224"+'.log', True)
    logger = logging.getLogger('./logs/'+"save_training_params_crossnet_224"+'.log')
    model_name = "resnet_multi_task"
    train_me_where = "from_beginning"

    mn_dataset_loader_multi_train, mn_dataset_loader_multi_valid = create_save_multi_dataset_together()

    num_classes = 2
    # batch_size = 8
    feature_extract = True

    is_use_cuda = False  # torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")

    my_model = VariousModels(model_name, num_classes, feature_extract)
    model_ft, input_size = my_model.initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 512)

    dd = .1

    model_1 = CombineMultiOutputModel(model_ft, dd)
    model_1 = model_1.to(device)

    epoch_num = "20"

    lrlast = .001
    lrmain = .0001

    optim1 = optim.Adam(
        [
            {"params": model_1.resnet_model.parameters(), "lr": lrmain},
            {"params": model_1.x11.parameters(), "lr": lrlast},
            {"params": model_1.x12.parameters(), "lr": lrlast},
            {"params": model_1.x13.parameters(), "lr": lrlast},
            {"params": model_1.x14.parameters(), "lr": lrlast},
            {"params": model_1.x21.parameters(), "lr": lrlast},
            {"params": model_1.x22.parameters(), "lr": lrlast},
            {"params": model_1.x23.parameters(), "lr": lrlast},
            {"params": model_1.x24.parameters(), "lr": lrlast},
            {"params": model_1.x1.parameters(), "lr": lrlast},
            {"params": model_1.x2.parameters(), "lr": lrlast},
            {"params": model_1.y1o.parameters(), "lr": lrlast},
            {"params": model_1.y2o.parameters(), "lr": lrlast},
            {"params": model_1.y3o.parameters(), "lr": lrlast},
            {"params": model_1.y4o.parameters(), "lr": lrlast},
        ])

    optimizer_ft = optim1  # Observe that all parameters are being optimized
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

    # Observe that all parameters are being optimized
    start_epoch = 0
    num_epochs = 90

    # Train and evaluate
    my_trainer = Trainer(model_1, optimizer_ft, exp_lr_scheduler, is_use_cuda,
                         mn_dataset_loader_multi_train, mn_dataset_loader_multi_valid, start_epoch, num_epochs, logger,model_name)
    my_trainer.train()


if __name__ == '__main__':
    main()
