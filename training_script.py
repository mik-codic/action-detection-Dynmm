# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import csv
import argparse
import copy
from datetime import datetime
import json
import pickle
import os
import sys
import time
import warnings
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn

from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model
from src import utils
from src.prepare_data import prepare_data
from src.utils import save_ckpt, save_ckpt_every_epoch
from src.utils import load_ckpt
from src.utils import print_log, ExpDecayTemp

from src.logger import CSVLogger
from src.confusion_matrix import ConfusionMatrixTensorflow, ConfusionMatrixPytorch, miou_pytorch
from src.datasets.xdviolance.pytorch_dataset_new import load_data,compute_weight_class
from src.datasets.xdviolance.preprocessing.data_stats import get_weights
import torchvision.transforms as transforms
import wandb 
def parse_args():
    parser = ArgumentParserRGBDSegmentation(
        description='Efficient violence recognition (Training)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    args = parser.parse_args()

    # The provided learning rate refers to the default batch size of 8.
    # When using different batch sizes we need to adjust the learning rate
    # accordingly:
    if args.batch_size != 8:
        args.lr = args.lr * args.batch_size / 8
        warnings.warn(f'Adapting learning rate to {args.lr} because provided '
                      f'batch size differs from default batch size of 8.')

    return args


def train_main():
    args = parse_args()

    # directory for storing weights and other training related files
    training_starttime = datetime.now().strftime("%d_%m_%Y-%H_%M_%S-%f")
    ckpt_dir = os.path.join(args.results_dir, args.dataset,
                            f'checkpoints_{training_starttime}')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, 'confusion_matrices'), exist_ok=True)

    with open(os.path.join(ckpt_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    with open(os.path.join(ckpt_dir, 'argsv.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')

    # when using multi scale supervision the label needs to be downsampled.
    label_downsampling_rates = [8, 16, 32]

    # data preparation ---------------------------------------------------------
    print('preparing data')
    #data_loaders = prepare_data(args, ckpt_dir)
    train_loader = load_data("/raid/home/dvl/mpresutto/vol/DynMM/FusionDynMM/datasets/xdviolence/training_final_prova")
    valid_loader = load_data("/raid/home/dvl/mpresutto/vol/DynMM/FusionDynMM/datasets/xdviolence/testing_final_prova")
    wandb.init()
    print("\nlen of training_loader",len(train_loader))
    # if args.valid_full_res:
    #     train_loader, valid_loader, valid_loader_full_res = data_loaders
    # else:
    #     train_loader, valid_loader = data_loaders
    #     valid_loader_full_res = None

    # n_classes_without_void = train_loader.dataset.n_classes_without_void
    # if args.class_weighting != 'None':
    #     class_weighting = train_loader.dataset.compute_class_weights(
    #         weight_mode=args.class_weighting,
    #         c=args.c_for_logarithmic_weighting)
    # else:
    #     class_weighting = np.ones(n_classes_without_void)
        
    # model building -----------------------------------------------------------
    print('building model')
    n_classes_without_void = 7

    model, device = build_model(args, n_classes=n_classes_without_void)
    # loss, optimizer, learning rate scheduler, csvlogger  ----------

    # loss functions (only loss_function_train is really needed.
    # The other loss functions are just there to compare valid loss to
    # train loss)
    #class_weighting = compute_weight_class(train_loader)
    loss_function_train = \
        utils.CrossEntropyLoss2d(weight=(0.16,0.16,0.16,0.16,0.16,0.16,0.16),device=device)
    class_weights = get_weights().float().cuda()
    loss_function_valid = nn.CrossEntropyLoss(weight=class_weights)
    loss_function_train_ = nn.CrossEntropyLoss()
    #loss_function_valid = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args, model)

    # in this script lr_scheduler.step() is only called once per epoch
    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=[i['lr'] for i in optimizer.param_groups],
        total_steps=args.epochs,
        div_factor=25,
        pct_start=0.1,
        anneal_strategy='cos',
        final_div_factor=1e4
    )

    # load checkpoint if parameter last_ckpt is provided
    if args.last_ckpt:
        ckpt_path = args.last_ckpt
        # ckpt_path = os.path.join(ckpt_dir, args.last_ckpt)
        epoch_last_ckpt, best_miou, best_miou_epoch = load_ckpt(model, optimizer, ckpt_path, device)
        start_epoch = epoch_last_ckpt + 1
    else:
        start_epoch = 0

    if args.freeze and args.dynamic:
        print('Freeze everything but the soft gates')
        model.freeze()

    valid_split = valid_loader

    # one confusion matrix per camera and one for whole valid data
    confusion_matrices = dict()
    ConfusionMatrixPytorch(n_classes_without_void)
    # for camera in cameras:
    #     confusion_matrices[camera] = ConfusionMatrixPytorch(n_classes_without_void)
    #     confusion_matrices['all'] = ConfusionMatrixPytorch(n_classes_without_void)
    
    # temperature scheduler
    temp_scheduler = ExpDecayTemp(start_t=args.temp, end_t=args.end_temp, time_len=args.epoch_hard)
    model.baseline = args.baseline
    # start training -----------------------------------------------------------
    print('start training')
    list_loss = []
    for epoch in range(int(start_epoch), args.epochs):
        assert args.epoch_ini <= args.epoch_hard
        model.ini_stage = True if epoch < args.epoch_ini else False
        model.hard_gate = True if epoch >= args.epoch_hard else False
        model.temp = temp_scheduler.get_t(epoch)

        logs = train_one_epoch(
            model, train_loader, device, optimizer, loss_function_train_, epoch,
            lr_scheduler, args.modality, label_downsampling_rates,
            loss_flop_ratio=args.loss_ratio, flop_budget=args.flop_budget, debug_mode=args.debug)
        list_loss.append(logs['loss_train_total'])
        np.savetxt("losses_list.csv", list_loss, newline="\n")

        print(f"Epoch {logs['epoch']} | Train loss {logs['loss_train_total']:.4f} | Flop loss {logs['loss_flop']:.4f} "
              f"Temperature {model.temp} | lr {logs['lr_0']}")
    #---------------------------------------------------------------          
        #validation after every epoch -----------------------------------------
        if epoch == start_epoch or epoch % args.eval_every == 0:
            miou = validate(
                model, valid_loader, device,
                confusion_matrices, args.modality, loss_function_valid, logs,
                ckpt_dir, epoch,
                debug_mode=args.debug, dynamic=args.dynamic, soft_eval=args.soft_eval
            )
            #print(f"Test loss {logs['loss_test']:.4f} | Test mIoU {logs['mIoU_test_kv1']:.4f} | "
            #      f"Best mIoU {best_miou:.4f} | Best mIoU epoch {best_miou_epoch}")
            print("\nresult\n",miou)
    #     if args.valid_full_res:
    #         miou_full_res, logs = validate(
    #             model, valid_loader_full_res, device,
    #             confusion_matrices, args.modality, loss_function_valid, logs,
    #             ckpt_dir,
    #             epoch, loss_function_valid_unweighted,
    #             add_log_key='_full-res', debug_mode=args.debug, dynamic=args.dynamic
    #         )

    #     logs.pop('time', None)
    #     csvlogger.write_logs(logs)

    #     # save weights
    #     # print(miou['all'])
    #     save_current_checkpoint = False
    #     if miou['kv1'] > best_miou:
    #         best_miou = miou['kv1']
    #         best_miou_epoch = epoch
    #         model_best = copy.deepcopy(model)
    #         # save_current_checkpoint = True

    #     if args.valid_full_res and miou_full_res['all'] > best_miou_full_res:
    #         best_miou_full_res = miou_full_res['all']
    #         best_miou_full_res_epoch = epoch
    #         save_current_checkpoint = True

    #     # don't save weights for the first 10 epochs as mIoU is likely getting
    #     # better anyway
        if epoch >= 5 and epoch % args.save_every == args.save_every - 1:
            save_ckpt(ckpt_dir, model, optimizer, epoch)

    #     # save / overwrite latest weights (useful for resuming training)
    #     # save_ckpt_every_epoch(ckpt_dir, model, optimizer, epoch, best_miou, best_miou_epoch)
    # save_ckpt(ckpt_dir, model_best, optimizer, best_miou_epoch)

    # # write a finish file with best miou values in order overview
    # # training result quickly
    # with open(os.path.join(ckpt_dir, 'finished.txt'), 'w') as f:
    #     f.write('best miou: {}\n'.format(best_miou))
    #     f.write('best miou epoch: {}\n'.format(best_miou_epoch))
    #     if args.valid_full_res:
    #         f.write(f'best miou full res: {best_miou_full_res}\n')
    #         f.write(f'best miou full res epoch: {best_miou_full_res_epoch}\n')
#--------------------------------------------------------
    print("Training completed ")


def train_one_epoch(model, train_loader, device, optimizer, loss_function_train,
                    epoch, lr_scheduler, modality,
                    label_downsampling_rates, loss_flop_ratio=0.0, flop_budget=0.0, debug_mode=False):
    global loss_flop
    loss_flop = torch.tensor(0)
    training_start_time = time.time()
    lr_scheduler.step(epoch)
    samples_of_epoch = 0

    # set model to train mode
    model.train()

    # loss for every resolution
    losses_list = []

    # summed loss of all resolutions
    total_loss_list = []
    loss_flop_list = []

    for i, (frame,spects,label) in tqdm(enumerate(train_loader)):
        

        print("\n\ntraining batch: ",i)
        if i == len(train_loader) - 1:
            model.start_weight()
        start_time_for_one_step = time.time()
        #print(len(frame))
        # load the data and send them to gpu
        # if modality in ['rgbd', 'rgb']:
        #     for sample in frame:
        #         sample = torch.from_numpy(np.asarray(sample))
        #         image = sample.to(device)

        #         image = np.transpose(sample, (0, 3, 1, 2))
        #         #print("\ndimensione immagine\n",image.size())
        #         image = image.to(device)
        #         image = image.float()

        #         # image = sample[0].to(device)
        #         # frames_ = image[0].cpu().numpy()
        #         # print(len(frames_))
        #         # im1 = Image.fromarray(frames_)
        #         # im1 = im1.save("geeks.jpg") 

        #         #print(frames_)
        #     for label_ in label:
        #         label_vec = torch.from_numpy(np.asarray(label_))
        #         #label_vec = np.transpose(label_vec,(1,0))

        #         label_vec = label_vec.to(device)
        #     for spec in spects:
        #         spect_ = torch.from_numpy(np.asarray(spec.cpu()))
        #         spect_ = spect_[:, None,:]
        #         #spect_ = np.transpose(spect_, (1, 0, 2, 3))
        #         #spect_ = spect_.resize()
        #         #print("\nsize spect\n",(spect_.size()))

        #         # sp = spect_[0].to(device)
        #         # #print(sp.shape)
        #         # sp = sp[0].cpu().numpy()
        #         # im1 = Image.fromarray(sp)
        #         # im1 = im1.convert('RGB')
        #         # im1 = im1.save("spect.jpg") 

        #         spect_ = spect_.to(device)
        #         spect = spect_.float()
        #         batch_size = image.data.shape[0]
        # if modality in ['rgbd', 'depth']:
        #     depth = frame.to(device)
        #     batch_size = depth.data.shape[0]
        #target_scales = [label.to(device)]
        # if len(label_downsampling_rates) > 0:
        #     for rate in sample['label_down']:
        #         target_scales.append(sample['label_down'][rate].to(device))

        optimizer.zero_grad()
        # this is more efficient than optimizer.zero_grad()
        # for param in model.parameters():
        #     param.grad = None

        # forward pass
        if modality == 'rgbd':
            #spect = torch.index_select(spect, 0, torch.tensor([i]).to(device))
            #print("\nimage len\n",len(image))
            #print("\spect len\n",len(spect))
            #print("dimensione spect e image\n\n",spect.size(),image.size())
            spe = spects.size()
            im_sz = frame.size()
            
            #print(spe[2],im_sz[2])
            #image = cv2.copyMakeBorder(image.detach().cpu().numpy(), 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0) 
            #transform = transforms.Pad((0,0,160, 50))
            #image = transform(image)
            #print(image.size())
            images = frame
            images = images.permute((0,3,1,2))
            images = images.float().cuda()
            #print(spects.size)
            
            #spect = spects.float()
            spect = spects.float().cuda()
            #spect = spect
            spect = spect[:, None,:,:]
            spect = spect.permute((0,1,3,2))
            print("\n\nDAMN\n\n",spect.size(),images.size())
            pred_scales,loss_flop = model(images, spect)

        elif modality == 'rgb':
            pred_scales = model(image)
        else:
            pred_scales = model(depth)

        # loss computation
        #label_vec = torch.squeeze(label_vec, dim=1)
        #print("\nvectors label\n",label_vec.size())
        #print("\nvectors prediction\n",pred_scales.size())
        class_weights = get_weights().float().cuda()
        loss_function_train_ = nn.CrossEntropyLoss(weight=class_weights) 
        label = torch.tensor([np.argmax(lab) for lab in label])
        losses = loss_function_train_(pred_scales, label.to(device))#+max(torch.zeros_like(loss_flop), loss_flop-flop_budget)
        print("\nloss: ",losses)
        loss_segmentation = sum([losses])
        l2 = max(torch.zeros_like(loss_flop), loss_flop-flop_budget)
        if loss_flop_ratio > 0:
            l2 = max(torch.zeros_like(loss_flop), loss_flop-flop_budget)
            # print(loss_segmentation.item(), loss_flop.item(), l2.item())
            total_loss = loss_segmentation + loss_flop_ratio * l2
        else:
            total_loss = loss_segmentation

        #total_loss.backward()
        losses.backward()
        optimizer.step()
        wandb.log({"training_loss":loss_segmentation})
        
        wandb.log({"flop loss": l2})

        # append loss values to the lists. Later we can calculate the
        # mean training loss of this epoch
        #losses_list.append([loss.cpu().detach().numpy() for loss in losses])
        
        
        losses_list.append(losses.cpu().detach().numpy())
        total_loss = total_loss.cpu().detach().numpy()
        total_loss_list.append(total_loss)
        if loss_flop_ratio > 0:
            loss_flop_list.append((loss_flop).cpu().detach().numpy())

        if np.isnan(total_loss):
            raise ValueError('Loss is None')

        # print log
        #samples_of_epoch += 3
        # time_inter = time.time() - start_time_for_one_step

        learning_rates = lr_scheduler.get_lr()

        # print_log(epoch, samples_of_epoch, batch_size,
        #           len(train_loader.dataset), total_loss, time_inter,
        #           learning_rates)

        if debug_mode:
            # only one batch while debugging
            break
        if i == len(train_loader) - 1:
            model.end_weight()
    
    # fill the logs for csv log file and web logger
    
    #wandb.log({"total_loss_list"})
    logs = dict()
    logs['time_training'] = time.time() - training_start_time
    logs['loss_train_total'] = np.mean(total_loss_list)
    logs['loss_flop'] = np.mean(loss_flop_list) if loss_flop_ratio > 0.0 else 0
    losses_train = np.mean(losses_list, axis=0)
    logs['loss_train_full_size'] = losses_train
    # for i, rate in enumerate(label_downsampling_rates):
    #     logs['loss_train_down_{}'.format(rate)] = losses_train[i + 1]
    logs['epoch'] = epoch
    for i, lr in enumerate(learning_rates):
        logs['lr_{}'.format(i)] = lr
    return logs


def validate(model, valid_loader, device, confusion_matrices,
             modality, loss_function_valid, logs, ckpt_dir, epoch,
             add_log_key='',
             debug_mode=False, dynamic=False, soft_eval=False):
    #valid_split = valid_loader.dataset.split + add_log_key
    valid_split = valid_loader
    # print(f'Validation on {valid_split}')

    # we want to track how long each part of the validation takes
    validation_start_time = time.time()
    cm_time = 0    # time for computing all confusion matrices
    forward_time = 0
    post_processing_time = 0
    copy_to_gpu_time = 0

    # set model to eval mode
    model.eval()
    model.hard_gate = False if soft_eval else True
    if dynamic:
        model.start_weight()

    # we want to store miou and ious for each camera
    
    acc = 0

    # reset loss (of last validation) to zero
    #loss_function_valid.reset_loss()

    # if loss_function_valid_unweighted is not None:
    #     loss_function_valid_unweighted.reset_loss()

    # validate each camera after another as all images of one camera have
    # the same resolution and can be resized together to the ground truth
    # segmentation size.
    
    
        #miou_th = miou_pytorch(confusion_matrices[camera])
        # confusion_matrices[camera].reset_conf_matrix()
        # print(f'{camera}: {len(valid_loader.dataset)} samples')
    total_count = 0
    total_acc = 0
    total_loss = 0
    for k, (frames,spects,labels) in tqdm(enumerate(valid_loader)):
        

        print("/nframe size:",frames.size())
        #print("\n\nbatch!!: ",i)
        # copy the data to gpu
        copy_to_gpu_time_start = time.time()
        if modality in ['rgbd', 'rgb']:
            images = frames.to(device)
        if modality in ['rgbd', 'depth']:
            spect = spects.to(device)
        if not device.type == 'cpu':
            torch.cuda.synchronize()
        copy_to_gpu_time += time.time() - copy_to_gpu_time_start

        # forward pass
        with torch.no_grad():
            forward_time_start = time.time()
            if modality == 'rgbd':
                images = images.permute((0,3,1,2)).cuda().float()
                spect = spect[:,None,:,:].cuda().float()
                spect = spect.permute((0,1,3,2))
                prediction = model(images, spect, test=True)
            elif modality == 'rgb':
                prediction = model(images)
            else:
                prediction = model(spect)
            if not device.type == 'cpu':
                torch.cuda.synchronize()
            forward_time += time.time() - forward_time_start

            # compute valid loss
            post_processing_time_start = time.time()

            predic = [torch.argmax(pred)for pred in prediction]
            
            labels = torch.tensor([np.argmax(lab) for lab in labels])
            loss_val = loss_function_valid(
                prediction,
                labels.to(device)
            )

            prediction = prediction.cpu()

            label = labels.numpy()
            prediction = prediction.numpy()
            #print(len(prediction))
            prediction_res = []
            for k in range(len(prediction)):
                prediction_res.append(np.argmax(prediction[k]))
            print(prediction_res)
            prediction_res = np.array(prediction_res)
            count = 0
            for j in range(0,len(labels)):
                print(prediction_res[j],labels[j])
                if prediction_res[j] == labels[j]:
                    count += 1
            print(count)
            acc = count/len(labels)
            loss_val = np.mean(loss_val.cpu().detach().numpy())
            total_count = total_count+1
            total_acc = total_acc+acc
            total_loss = total_loss + loss_val
            print("batch_accuracy: ", acc)
            
            post_processing_time += \
                time.time() - post_processing_time_start

            # finally compute the confusion matrix
            cm_start_time = time.time()
            #confusion_matrices.update(torch.from_numpy(label), torch.from_numpy(prediction))
            # confusion_matrices[camera].update_conf_matrix(label, prediction)
            cm_time += time.time() - cm_start_time

            if debug_mode:
                # only one batch while debugging
                break

        # After all examples of camera are passed through the model,
        # we can compute miou and ious.
        cm_start_time = time.time()
        # miou[camera] = miou_th.compute().data.numpy()
        # ious[camera] = 0
        cm_time += time.time() - cm_start_time
        # print(f'mIoU {valid_split} {camera}: {miou[camera]}')

    # confusion matrix for the whole split
    # (sum up the confusion matrices of all cameras)
    # cm_start_time = time.time()
    # confusion_matrices['all'].reset()
    # confusion_matrices['all'].reset_conf_matrix()
    # for camera in cameras:
    #     confusion_matrices['all'].confusion_matrix += \
    #         confusion_matrices[camera].confusion_matrix.numpy()

    # miou and iou for all cameras
    # miou['all'], ious['all'] = confusion_matrices['all'].compute_miou()
    # cm_time += time.time() - cm_start_time

    if dynamic:
        model.end_weight(print_each=True)
    # print('weight', model.end_weight(print_each=True))
    # print(f"mIoU {valid_split}: {miou['all']}")

    validation_time = time.time() - validation_start_time

    # save the confusion matrices of this epoch.
    # This helps if we want to compute other metrics later.
    with open(os.path.join(ckpt_dir, 'confusion_matrices',
                           f'cm_epoch_{epoch}.pickle'), 'wb') as f:
        pickle.dump({k: cm.confusion_matrix.numpy()
                     for k, cm in confusion_matrices.items()}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
        # pickle.dump({k: cm.overall_confusion_matrix
        #              for k, cm in confusion_matrices.items()}, f,
        #             protocol=pickle.HIGHEST_PROTOCOL)

    # logs for the csv logger and the web logger
    # logs[f'loss_{valid_split}'] = \
    #     loss_function_valid.compute_whole_loss()

    # if loss_function_valid_unweighted is not None:
    #     logs[f'loss_{valid_split}_unweighted'] = \
    #         loss_function_valid_unweighted.compute_whole_loss()

    # logs[f'mIoU_{valid_split}'] = miou['all']
    # for camera in cameras:
    #     logs[f'mIoU_{valid_split}_{camera}'] = miou[camera]
    wandb.log({"val_loss": total_loss/len(valid_loader)})
    wandb.log({"accuracy":total_acc/len(valid_loader)})
    logs['time_validation'] = validation_time
    logs['time_confusion_matrix'] = cm_time
    logs['time_forward'] = forward_time
    logs['time_post_processing'] = post_processing_time
    logs['time_copy_to_gpu'] = copy_to_gpu_time

    # write iou value of every class to logs
    # for i, iou_value in enumerate(ious['all']):
    #     logs[f'IoU_{valid_split}_class_{i}'] = iou_value

    return np.mean(total_acc)


def get_optimizer(args, model):
    # set different learning rates fo different parts of the model
    # when using default parameters the whole model is trained with the same
    # learning rate
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=True
        )
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    else:
        raise NotImplementedError(
            'Currently only SGD and Adam as optimizers are '
            'supported. Got {}'.format(args.optimizer))

    print('Using {} as optimizer'.format(args.optimizer))
    return optimizer


if __name__ == '__main__':
    train_main()


