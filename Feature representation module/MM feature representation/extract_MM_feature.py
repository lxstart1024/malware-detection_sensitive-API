import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from vit_model import vit_base_patch16_224_in21k as create_model
import numpy as np
import csv

def MM_feature_extractor(path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # create model
    model = create_model(num_classes=5, has_logits=False).to(device)
    # load model weights
    model_weight_path = "./weights/model-9.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    directs = os.listdir(path)
    for app_name in directs:
        app_image = path + "/" + app_name + "/" + "Grayscale"
        app_image_file = os.listdir(app_image)
        with open(app_name + ".csv", mode="a", newline="") as f_csv:
            writer = csv.writer(f_csv)
            for file_name in app_image_file:
                file_path = app_image + "/" + file_name
                assert os.path.exists(file_path), "file: '{}' dose not exist.".format(file_path)
                img = Image.open(file_path)
                img = data_transform(img)
                img = torch.unsqueeze(img, dim=0)
                writer.writerow([model.forward_features(img.to(device)).tolist()[0]])



if __name__ == '__main__':
    path = "F:/MM_DATA"
    MM_feature_extractor(path)