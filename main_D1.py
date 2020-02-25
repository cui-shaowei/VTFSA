import torch
import torch.nn as nn
import os
import shutil
import time
import torch.nn.parallel

from libs.dataset.D1_dataloader import TsingHuaVTDataset
from utils import Bar, Logger, AverageMeter, savefig, ACC
# from models.models import *
# from vtf_mhsa import *
import cv2
from network_utils import init_weights_xavier
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from torch.utils import data
import argparse
import random
import torch.backends.cudnn as cudnn
from utils import mkdir_p
import torchvision.transforms as transforms

from modules.vtfsa_freeze import *

parser = argparse.ArgumentParser(description='Visual-Tactile Fusion Transformer for slip detection')
parser.add_argument('-f', default='', type=str)

# Tuning
parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')

# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--name', type=str, default='vtf-transformer',
                    help='name of the trial (default: "vtf-transformer")')
parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: \
                                e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar= \
    'PATH', help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                         help='path to latest checkpoint (default: none)')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
args.use_cuda = torch.cuda.is_available()
str_ids = args.gpu_ids.split(',')
args.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        args.gpu_ids.append(id)
# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
    cudnn.benchmark = True
    cudnn.enabled = True


####################################################################



hyp_params.model = 'VTF_SA_CNNs'
hyp_params.output_dim = 2
hyp_params.batch_size = 64
hyp_params.lr=1e-3
hyp_params.res_size=224
hyp_params.num_epochs=1000
# hyp_params.gpu_ids='0,1,2,7'
transform = transforms.Compose([transforms.Resize([hyp_params.res_size, hyp_params.res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def main():

    # start_epoch = opt.start_epoch  # start from epoch 0 or last checkpoint epoch
    # opt.phase = 'train'
    print("Start loading the data....")
    trainset = TsingHuaVTDataset("../Visual-Tactile_Dataset/dataset", train=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=hyp_params.batch_size,
        shuffle=True,
        num_workers=8
    )
    # opt.phase = 'val'
    validset = TsingHuaVTDataset("../Visual-Tactile_Dataset/dataset", train=False)
    val_loader = torch.utils.data.DataLoader(
        dataset=validset,
        batch_size=hyp_params.batch_size,
        shuffle=False,
        num_workers=8
    )

    print('Finish loading the data....')
    hyp_params.n_train, hyp_params.n_test = len(train_loader), len(val_loader)

    model_pre=VTF_CNNs_pre_freeze()
    model_fcs=VTF_SA()
    model_pre = torch.nn.DataParallel(model_pre).cuda()
    model_fcs = torch.nn.DataParallel(model_fcs).cuda()
    print('Total params: %.2fM' % (sum(p.numel() for p in model_fcs.parameters()) / 1000000.0))
    print('LR:',hyp_params.lr)
    print('Batch_size:', hyp_params.batch_size)

    optimizer = torch.optim.Adam(model_fcs.parameters(), lr=hyp_params.lr, betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=0)

    model_fcs.apply(init_weights_xavier)

    title = hyp_params.model
    if hyp_params.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(hyp_params.resume), 'Error: no checkpoint directory found!'
        hyp_params.checkpoint = os.path.dirname(hyp_params.resume)
        checkpoint = torch.load(hyp_params.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model_fcs.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(hyp_params.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(hyp_params.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Valid Acc.'])

    expr_dir = os.path.join(hyp_params.checkpoint, hyp_params.name)
    mkdir_p(expr_dir)
    file_name = os.path.join(expr_dir, 'opt.txt')
    # with open(file_name, 'wt') as opt_file:
    #     opt_file.write('------------ Options -------------\n')
    #     for k, v in sorted(args.items()):
    #         opt_file.write('%s: %s\n' % (str(k), str(v)))
    #     opt_file.write('-------------- End ----------------\n')

    # Train and val

    best_acc = 0
    for epoch in range(hyp_params.num_epochs):
        if epoch <5:
            hyp_params.lr=1e-3
            optimizer = torch.optim.Adam(model_fcs.parameters(), lr=hyp_params.lr, betas=(0.9, 0.999), eps=1e-08,
                                         weight_decay=0)
        elif epoch>5 and epoch<10:
            hyp_params.lr=1e-4
            optimizer = torch.optim.Adam(model_fcs.parameters(), lr=hyp_params.lr, betas=(0.9, 0.999), eps=1e-08,
                                         weight_decay=0)
        elif epoch > 9 and epoch < 20:
            hyp_params.lr = 1e-4
            optimizer = torch.optim.Adam(model_pre.parameters(), lr=hyp_params.lr, betas=(0.9, 0.999), eps=1e-08,
                                         weight_decay=0)
        elif epoch>19:
            hyp_params.lr = 1e-4
            crnn_params = list(model_pre.parameters())+list(model_fcs.parameters())
            optimizer = torch.optim.Adam(crnn_params, lr=hyp_params.lr, betas=(0.9, 0.999), eps=1e-08,
                                         weight_decay=0)
        # adjust_learning_rate(optimizer, epoch, opt)
        # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50, verbose=False,
        #                                            threshold=0.0001, threshold_mode='rel', cooldown=20, min_lr=0,
        #                                            eps=1e-08)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, hyp_params.num_epochs, hyp_params.lr))

        train_loss = train(train_loader, model_pre,model_fcs, optimizer, epoch)
        test_loss, test_acc,pred,targets= valid(val_loader, model_pre,model_fcs, epoch)

        # append logger file
        logger.append([hyp_params.lr, train_loss, test_loss, test_acc])
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        # save model
        save_dir = 'VTFSA_results/checkpoint'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict_pre': model_pre.state_dict(),
            'state_dict_fcs': model_fcs.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=save_dir, pred=pred, targets=targets)
        print('Best acc:',best_acc)
        # print(best_acc)
    # acc_list.append(best_acc)

def train(trainloader, model_pre,model_fcs, optimizer, epoch):
    # switch to train mode
    model_fcs.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (rgb,gel,label) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        # x_tactile,x_visual, targets = torch.autograd.Variable(x_tactile),torch.autograd.Variable(x_visual), torch.autograd.Variable(targets)
            # inputs = inputs.cuda()
        # x_tactile=x_tactile.cuda()
        # x_visual=x_visual.cuda()
        # targets = targets.cuda(non_blocking=True)
        rgb,gel,label=rgb[1].cuda(),gel.cuda(),label.cuda()
        # print(x_tactile.shape)
        # compute output
        rgb = model_pre(rgb)
        outputs = model_fcs(rgb,gel)
        loss = F.cross_entropy(outputs, label)
        y_pred = torch.max(outputs, 1)[1]
        acc =  accuracy_score(y_pred.cpu().data.numpy(), label.cpu().data.numpy())

        # measure the result
        losses.update(loss.item(), rgb.size(0))
        avg_acc.update(acc, rgb.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}| ACC(input): {acc: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    acc=avg_acc.avg,
                    )
        bar.next()
    bar.finish()
    return losses.avg


def valid(testloader, model_pre,model_fcs, epoch):
    # switch to train mode
    model_fcs.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()
    end = time.time()
    preds = []
    targets_list = []
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (rgb,gel, label) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        rgb, gel, label = rgb[1].cuda(), gel.cuda(), label.cuda()
        # print(x_tactile.shape)
        # compute output
        rgb = model_pre(rgb)
        outputs = model_fcs(rgb,gel)


        loss = F.cross_entropy(outputs, label)
        y_pred = torch.max(outputs, 1)[1]
        for i in range(outputs.size(0)):
            preds.append(y_pred[i].cpu().data.numpy())
            targets_list.append(label[i].cpu().data.numpy())
        acc = accuracy_score(y_pred.cpu().data.numpy(), label.cpu().data.numpy())

        # measure the result
        losses.update(loss.item(), rgb.size(0))
        avg_acc.update(acc, rgb.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}| ACC(input): {acc: .4f}'.format(
            batch=batch_idx + 1,
            size=len(testloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            acc=avg_acc.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, avg_acc.avg,preds,targets_list)


def adjust_learning_rate(optimizer, epoch, opt):
    if epoch % opt.schedule ==0 and epoch !=0 :
        opt.lr *= opt.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.lr


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar',pred=[],targets=[]):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
        np.save(checkpoint+'/best_pred.npy',pred)
        np.save(checkpoint+'/best_targets.npy',targets)


if __name__ == "__main__":
    main()