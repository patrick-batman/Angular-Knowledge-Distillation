import os
import pandas as pd
from PIL import Image

# MTCNN


from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Gives easier dataset managment and creates mini batches


class PeopleDataset(Dataset):
    def __init__(self, csv_file, root_dir, default_label, default_img ,transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.default_label = default_label
        self.default_img = default_img
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path =  os.path.join(self.root_dir,self.annotations.iloc[index, 0]) 
        image = Image.open(img_path)
        img_cropped = mtcnn(image)
        y_label = self.annotations.iloc[index,1]

        # if self.transform:
        #     image = self.transform(image)
        if(img_cropped == None):
            return self.default_img,self.default_label

        return img_cropped, y_label


from facenet_pytorch import MTCNN
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device="cpu"
)

root_dir ="CASIA_dataset/Images"

img = Image.open(os.path.join(root_dir,'0.png'))
label_default = 4250
# Get cropped and prewhitened image tensor
img_cropped_default = mtcnn(img)


# Load Data
dataset = PeopleDataset(
    csv_file="CASIA_dataset/annotations.csv",
    root_dir=root_dir,
    default_label=label_default,
    default_img=img_cropped_default
)
# DATA LOADER
def create_train_loader(batch_size , num_workers):
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    return train_loader


