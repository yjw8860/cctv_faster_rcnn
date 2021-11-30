import os
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset

class cctvTestDataset(Dataset):
    def __init__(self, img_folder_path):
        self.img_list = os.listdir(img_folder_path)
        self.img_list = [os.path.join(img_folder_path, img_filename) for img_filename in self.img_list]
        self.to_tensor = T.ToTensor()

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.img_list[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.to_tensor(img)

        return img

    def __len__(self):
        return len(self.img_list)