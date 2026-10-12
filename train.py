import argparse
import os
import numpy as np
import cv2
from glob import glob

import torch

from utils import clip_gradient, AvgMeter
from torch.autograd import Variable
from datetime import datetime
import torch.nn.functional as F

from models.F2ANet import F2ANet
import albumentations as A


class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_paths, mask_paths, train_size, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.train_size = train_size
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = cv2.resize(image, (self.train_size, self.train_size))
        mask = cv2.resize(mask, (self.train_size, self.train_size))

        image = image.astype('float32') / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:, :, np.newaxis]
        mask = mask.astype('float32') / 255
        mask = mask.transpose((2, 0, 1))

        return np.asarray(image), np.asarray(mask)


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (wbce * weit).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def adjust_lr(optimizer, init_lr, epoch, args):
    if epoch % args.decay_epoch == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.decay_rate * init_lr
            lr = param_group['lr']
        return lr
    else:
        return optimizer.param_groups[0]['lr']


def train(train_loader, model, optimizer, epoch, cur_lr, args):
    model.train()

    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()

    with torch.autograd.set_detect_anomaly(True):
        for i, pack in enumerate(train_loader, start=1):
            for rate in size_rates:
                optimizer.zero_grad()

                # ---- data prepare ----
                images, gts = pack
                images = Variable(images)
                gts = Variable(gts)

                # ---- rescale ----
                train_size = int(round(args.train_size * rate / 32) * 32)
                images = F.interpolate(images, size=(train_size, train_size), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(train_size, train_size), mode='bilinear', align_corners=True)

                images = images.cuda()
                gts = gts.cuda()

                # ---- forward ----
                map4, map3, map2, map1 = model(images)

                map1 = F.interpolate(map1, size=(train_size, train_size), mode='bilinear', align_corners=True)
                map2 = F.interpolate(map2, size=(train_size, train_size), mode='bilinear', align_corners=True)
                map3 = F.interpolate(map3, size=(train_size, train_size), mode='bilinear', align_corners=True)
                map4 = F.interpolate(map4, size=(train_size, train_size), mode='bilinear', align_corners=True)

                # ---- loss ----
                loss = (
                    structure_loss(map1, gts)
                    + structure_loss(map2, gts)
                    + structure_loss(map3, gts)
                    + structure_loss(map4, gts)
                )

                # ---- backward ----
                loss.backward()
                clip_gradient(optimizer, args.clip)
                optimizer.step()

                # ---- recording loss ----
                if rate == 1:
                    loss_record.update(loss.data, args.batch_size)

            # ---- train visualization ----
            if i % 20 == 0 or i == total_step:
                print('{} Epoch [{:03d}/{:03d}], Iteration [{:03d}/{:03d}], Loss [{:0.4f}]'.format(
                    datetime.now(), epoch, args.num_epochs, i, total_step, loss_record.show()
                ))

    ckpt_path = save_path + str(epoch) + '_Polyp.pth'
    print('Saving Checkpoint', '[{}]'.format(ckpt_path))
    checkpoint = {
        'epoch': epoch + 1,
        'cur_lr': cur_lr,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, ckpt_path)
    return checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int,
                        default=120, help='epoch number')
    parser.add_argument('--backbone', type=str,
                        default='b3', help='backbone version')
    parser.add_argument('--init_lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=40, help='every n epochs decay learning rate')
    parser.add_argument('--batch_size', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--train_size', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default='./polyp_datasets/TrainDataset', help='path to train dataset')
    parser.add_argument('--resume_path', type=str,
                        default='', help='path to checkpoint for resume training')

    args = parser.parse_args()

    save_path = './snapshots'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    else:
        print('Save path existed')

    train_img_paths = []
    train_mask_paths = []
    train_img_paths = glob(os.path.join(args.train_path, 'Imgs', '*'))
    train_mask_paths = glob(os.path.join(args.train_path, 'GT', '*'))
    train_img_paths.sort()
    train_mask_paths.sort()

    transform = A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    train_dataset = Dataset(train_img_paths, train_mask_paths, args.train_size, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    total_step = len(train_loader)

    model = F2ANet(pretrained='./pretrained/mit_{}.pth'.format(args.backbone)).cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.init_lr)

    start_epoch = 1
    cur_lr = args.init_lr

    if args.resume_path != '':
        checkpoint = torch.load(args.resume_path)
        start_epoch = checkpoint['epoch']
        cur_lr = checkpoint['cur_lr']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(start_epoch, args.num_epochs + 1):
        cur_lr = adjust_lr(optimizer, cur_lr, epoch, args)
        checkpoint = train(train_loader, model, optimizer, epoch, cur_lr, args)
