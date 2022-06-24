import numpy as np
import torch
import glob
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from data import ImageNetDataset, get_transform, get_inv_transform
from model import TestModel
from tqdm import tqdm
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import resnet


class Metric():
    def __init__(self, num_classes, device="cuda"):
        self.metric_collection = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(),
            torchmetrics.Precision(num_classes=num_classes, average='macro'),
            torchmetrics.Recall(num_classes=num_classes, average='macro')
        ])
        if device == "cuda":
            self.metric_collection = self.metric_collection.to(
                torch.device('cuda:0'))

    @torch.no_grad()
    def __call__(self, output, label):
        return self.metric_collection(output, label)

    @torch.no_grad()
    def compute(self):
        return self.metric_collection.compute()

    def reset(self):
        self.metric_collection.reset()


def load_my_weight(model):
    params = model.state_dict()
    prefix = "../weights/"
    suffix = ".npy"
    for para in params:
        if "num_batches_tracked" in para:
            continue
        name: str = para.replace(".", "-")
        pos = name.find("-")
        if "layer" in name:
            name = name[:pos+1]+"block" + \
                str(int(name[pos+1])+1) + name[pos+2:]
        name = name.replace("downsample-0", "conv_ds")
        name = name.replace("downsample-1", "bn_ds")

        # if prefix+name+suffix not in weights_paths:
        #     print(name)

        npy: np.ndarray = np.load(prefix+name+suffix)
        if tuple(params[para].shape) != npy.shape:
            if "fc-weight" == name:
                npy = npy.reshape(npy.shape[-2], npy.shape[-1])
                npy = npy.transpose((1, 0))
            else:
                assert(len(tuple(params[para].shape)) == 1)
                npy = npy.reshape(-1)

        # if tuple(params[para].shape) != npy.shape:
        #     print(name)
        #     print(params[para].shape)
        #     print(npy.shape)

        params[para] = torch.Tensor(npy)
    return params


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    # torch.manual_seed(1118)
    # import random
    # random.seed(1118)
    # import numpy as np
    # np.random.seed(1118)

    train_dataset = ImageNetDataset("../ImageNet", "train", get_transform())
    val_dataset = ImageNetDataset("../ImageNet", "val", get_transform())

    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = True
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=64, shuffle=False, drop_last=True, num_workers=12)
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=12)

    # model = torchvision.models.resnet18(
    #     pretrained=False, zero_init_residual=True, num_classes=10)
    model = resnet.resnet18(
        pretrained=False, zero_init_residual=True, num_classes=10)
    params = load_my_weight(model)
    model.load_state_dict(params, strict=False)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.32, momentum=0)
    loss_func = nn.CrossEntropyLoss()
    model = model.cuda()
    writer = SummaryWriter()

    metric_method = Metric(num_classes=10)

    total_epoch = 25
    step = 0
    for epoch in range(total_epoch):
        print(f"Running at Epoch [{epoch+1}/{total_epoch}]")
        model.train()
        for img, label in tqdm(train_dataloader):
            step += 1
            img = img.cuda()
            label = label.cuda()
            output = model(img)
            loss = loss_func(output, label)
            print(loss.item())
            batch_metric = metric_method(output, label)
            # print(output[0])

            optimizer.zero_grad()
            loss.backward()
            # if step == 1:
            # np.save("../weight_converter/fc_weight_grad_std.npy",
            #         model.fc.weight.grad.cpu().detach().numpy())
            # exit(0)
            optimizer.step()
            # if step == 1:
            #     np.save("../weight_converter/layer4.0.downsample.0.weight_grad.npy",
            #             model.layer4[0].downsample[0].weight.grad.cpu().detach().numpy())
            #     exit(0)
            # if step == 1:
            #     torch.save(model.state_dict(), "step1.pt")
            #     exit(0)
            # print(loss.item())
            # for i in range(10):
            #     print(model.conv1.weight.grad.reshape(-1)[1024+i].item())
            # exit()
            writer.add_scalar(
                "train_loss", scalar_value=loss.item(), global_step=step)

        print(f"My metric:{metric_method.compute()}")
        metric_method.reset()

        confusion_matrix = np.zeros((10, 10))
        model.eval()
        for img, label in tqdm(val_dataloader):
            step += 1
            img = img.cuda()
            label = label.cuda()
            output = model(img)
            loss = loss_func(output, label)
            batch_metric = metric_method(output, label)
            predict = output.cpu().argmax(dim=-1)
            for i in range(label.shape[0]):
                confusion_matrix[predict[i], label[i]] += 1

        # print(confusion_matrix)
        metric_result = metric_method.compute()
        metric_method.reset()
        print(f"My metric: {metric_result}")
        writer.add_scalar(
            "val acc", scalar_value=metric_result["Accuracy"].item(), global_step=epoch)
