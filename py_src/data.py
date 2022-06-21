from cProfile import label
import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision


class ImageNetDataset(Dataset):
    def __init__(self, root, run_type, transform) -> None:
        super().__init__()
        assert(run_type == "train" or run_type == "val")
        self.img_path_label_pairs = []
        self.transform = transform

        category_paths = glob.glob(root+"/"+run_type+"/*")
        category_index_mapping = {}

        with open(root+"/"+"index2folder.txt") as fin:
            lines = fin.read().split("\n")[1:-1]
            for line in lines:
                vals = line.split(" ")
                category_index_mapping[vals[1]] = int(vals[0])

        for category_path in category_paths:
            category_name = category_path.split("/")[-1]
            img_paths = glob.glob(category_path+"/*")
            for img_path in img_paths:
                if category_index_mapping[category_name] % 100 == 0:
                    self.img_path_label_pairs.append(
                        [img_path, category_index_mapping[category_name]//100])
                # self.img_path_label_pairs.append(
                #     [img_path, category_index_mapping[category_name]])

    def __len__(self):
        return len(self.img_path_label_pairs)

    def __getitem__(self, index):
        img = Image.open(self.img_path_label_pairs[index][0])
        label = self.img_path_label_pairs[index][1]
        img = self.transform(img)
        return img, label


class ImageNetDatasetMini(Dataset):
    def __init__(self, root, run_type, transform) -> None:
        super().__init__()
        assert(run_type == "train" or run_type == "val")
        self.img_path_label_pairs = []
        self.transform = transform

        category_paths = glob.glob(root+"/"+run_type+"/*")
        category_index_mapping = {}

        with open(root+"/"+"index2folder.txt") as fin:
            lines = fin.read().split("\n")[1:-1]
            for line in lines:
                vals = line.split(" ")
                category_index_mapping[vals[1]] = int(vals[0])

        for category_path in category_paths:
            category_name = category_path.split("/")[-1]
            img_paths = glob.glob(category_path+"/*")
            for img_path in img_paths:
                self.img_path_label_pairs.append(
                    [img_path, category_index_mapping[category_name]])

    def __len__(self):
        return len(self.img_path_label_pairs)

    def __getitem__(self, index):
        img = Image.open(self.img_path_label_pairs[index][0])
        label = self.img_path_label_pairs[index][1]
        img = self.transform(img)
        return img, label


class ToRGB:
    def __call__(self, x):
        return x.convert("RGB")


def get_transform():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        ToRGB(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    return transform


def get_inv_transform():
    inv_transform = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0., 0., 0.],
                                                                                     std=[1/0.229, 1/0.224, 1/0.225]),
                                                    torchvision.transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                                                     std=[1., 1., 1.]),
                                                    torchvision.transforms.ToPILImage()
                                                    ])
    return inv_transform
