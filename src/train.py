from Cityscapes import Cityscapes
from utils import *
from model import get_model_instance_segmentation
from pytorch_lightning import Trainer
import torchvision.utils
from engine import train_one_epoch, evaluate
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

dataset = Cityscapes("C:\projects\cityscapes_data_preparation\data\\final\instances", ids_to_labels, 4,
                     get_transform(True))
indices = torch.randperm(len(dataset)).tolist()
train_dataset = torch.utils.data.Subset(dataset, indices[0:2])
test_dataset = torch.utils.data.Subset(dataset, indices[2:])
train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle=True, num_workers=1,
    collate_fn=collate_fn)
test_data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=True, num_workers=1,
    collate_fn=collate_fn)

device = torch.device("cpu")
num_classes = 3
model = get_model_instance_segmentation(num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

if __name__ == '__main__':
    num_epochs = 1
    for epoch in range(num_epochs):

        metrics = train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=1).__str__()
        print(metrics)
        # print(metrics)
        lr_scheduler.step()
# evaluate on the test dataset
        evaluate(model, test_data_loader, device=device)
        # update the learning rate
        # lr_scheduler.step()
        # # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device)
        # lr: 0.005000  loss: 3.2101 (4.9608)  loss_classifier: 1.0499 (0.8729)  loss_box_reg: 0.2224 (0.2056)  loss_mask: 1.5728 (1.6442)  loss_objectness: 0.2880 (1.8971)  loss_rpn_box_reg: 0.0328 (0.3410)
        #                     median global avg
