import argparse

import time

from copy import deepcopy

from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import pdb
import yaml
import torchvision
from dotmap import DotMap
import logging

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip import get_coop
from clip.cocoop import get_cocoop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from data.video import Video_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed, init_distributed_mode
from data.transforms import GroupScale, GroupCenterCrop, Stack, ToTorchFormatTensor, GroupNormalize, GroupOverSample, GroupFullResSample
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def test_time_tuning(model, inputs, optimizer, scaler, args, config):
    if args.cocoop:
        image_feature, pgen_ctx = inputs
        pgen_ctx.requires_grad = True
        optimizer = torch.optim.AdamW([pgen_ctx], args.lr)
    
    selected_idx = None


    for name, param in model.named_parameters():
        #only require parameter for the prompt
        if "prompt_learner" in name:
            param.requires_grad_(True)

    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():
            if args.cocoop:
                output = model((image_feature, pgen_ctx))
            else:
                output = model(inputs, config.data.num_segments)  #inputs torch.Size([24, 3, 224, 224]), outputs torch.Size([24, 400])
            output, selected_idx = select_confident_samples(output, args.selection_p)
                #torch.Size([48, 400]) , 3 view x 16 temporal variation 
            # if selected_idx is not None:
            #     output = output[selected_idx]
            # else:
            #     output, selected_idx = select_confident_samples(output, args.selection_p)
            #     #output torch.Size([6, 102]), selection_p = 0.1, select 10% from 64 samples

            #calibrate
            loss = avg_entropy(output)

        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()
        

        
        #calibrate (kl_divergence)
        
        # W = np.linalg.inv(np.identity(config.data.num_classes) * p_cf.detach().cpu().numpy())
        # b = np.zeros([config.data.num_classes, 1])
        # label_probs = output[0].detach().cpu().numpy()
        # label_probs = label_probs / np.sum(label_probs) #normalize to 1
        # calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b
        # calibrate_label_probs = torch.from_numpy(calibrate_label_probs).cuda(args.gpu).reshape(1, -1)
        # acc1, acc5 = accuracy(calibrate_label_probs, target, topk=(1, 5))
    if args.cocoop:
        return pgen_ctx

    return


def main():
    args = parser.parse_args()
    set_random_seed(args.seed)

    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    logging.basicConfig(filename=args.logging, level=logging.INFO, format='%(asctime)s: %(message)s')
    set_random_seed(args.seed)
    init_distributed_mode(args)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = DotMap(config)
    
    # control the spatial crop
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    input_size = config.data.input_size
    scale_size = 256 if config.data.input_size == 224 else config.data.input_size
    if args.test_crops == 1: # one crop
        cropping = torchvision.transforms.Compose([
            GroupScale(scale_size),
            GroupCenterCrop(input_size),
        ])
    elif args.test_crops == 3:  # do not flip, so only 3 crops (left right center)
        cropping = torchvision.transforms.Compose([
            GroupFullResSample(
                crop_size=input_size,
                scale_size=scale_size,
                flip=False)
        ])
    elif args.test_crops == 5:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupOverSample(
                crop_size=input_size,
                scale_size=scale_size,
                flip=False)
        ])

    logging.info("Use GPU: {} for training".format(args.gpu))

    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
    with open(config.data.label_list) as f: classnames = [x.strip().split(',')[1] for x in f.readlines()[1:]]
    if args.cocoop:
        model = get_cocoop(args.arch, args.test_sets, 'cpu', args.n_ctx)
        assert args.load is not None
        load_model_weight(args.load, model, 'cpu', args) # to load to cuda: device="cuda:{}".format(args.gpu)
        model_state = deepcopy(model.state_dict())
    else:
        # checkpoint = torch.load('/media/ssd8T/TPT-video/checkpoint/vificlip/hmdb51_seed3_vifi_clip_base2novel.pth')
        checkpoint = torch.load('/media/ssd8T/TPT-video/checkpoint/vificlip/vifi_clip_10_epochs_k400_full_finetuned.pth')
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        # new_state_dict["prompt_learner.token_suffix"].shape = torch.Size([400, 76, 512])
        # model.state_dict()["prompt_learner.ctx"].shape = [3, 512]
        # neutral_model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init, neutral_classnames)
        
        model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init, classnames) #load this model for tpt
        from fvcore.nn import FlopCountAnalysis
        pdb.set_trace()
        flops = FlopCountAnalysis(model, input)
        flops.total()

        filtered_state_dict = {}
        ignore_key = 'prompt_learner'
        for name, param in new_state_dict.items():
            if ignore_key not in name:
                filtered_state_dict[name] = param
        model.load_state_dict(filtered_state_dict, strict = False)
        if args.load is not None: #false for tpt will not go inside here
            logging.info("Use pre-trained soft prompt (CoOp) as initialization")
            pretrained_ctx = torch.load(args.load)['state_dict']['ctx']
            assert pretrained_ctx.size()[0] == args.n_ctx
            with torch.no_grad():
                model.prompt_learner[0].ctx.copy_(pretrained_ctx)
                model.prompt_learner[0].ctx_init_state = pretrained_ctx
        model_state = None

    for name, param in model.named_parameters():
        #only require parameter for the prompt
        if not args.cocoop:
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        elif "text_encoder" not in name:
                param.requires_grad_(False)
    
    logging.info("=> Model created: visual backbone {}".format(args.arch))
    
    if not torch.cuda.is_available():
        logging.info('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # define optimizer
    if args.cocoop:
        optimizer = None
        optim_state = None
    else:
        trainable_param = model.prompt_learner.parameters()
        optimizer = torch.optim.AdamW(trainable_param, args.lr)
        optim_state = deepcopy(optimizer.state_dict())
    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    logging.info('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True

    # norm stats from clip.load()
    # normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
    #                                  std=[0.26862954, 0.26130258, 0.27577711])

    
    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    results = {}

    set_id = 'kinetics400'
    # for set_id in datasets:
    #     if args.tpt:
    #         base_transform = transforms.Compose([
    #             transforms.Resize(args.resolution, interpolation=BICUBIC),
    #             transforms.CenterCrop(args.resolution)])
    #         preprocess = transforms.Compose([
    #             transforms.ToTensor(),
    #             normalize])
    #         data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
    #                                         augmix=len(set_id)>1)
    #         batchsize = 1
    #     else:
    #         data_transform = transforms.Compose([
    #             transforms.Resize(args.resolution, interpolation=BICUBIC),
    #             transforms.CenterCrop(args.resolution),
    #             transforms.ToTensor(),
    #             normalize,
    #         ])
    #         batchsize = args.batch_size

    #     logging.info("evaluating: {}".format(set_id))
    #     # reset the model
    #     # Reset classnames of custom CLIP model
    #     if len(set_id) > 1: 
    #         # fine-grained classification datasets
    #         classnames = eval("{}_classes".format(set_id.lower()))
    #     else:
    #         assert set_id in ['A', 'R', 'K', 'V', 'I']
    #         classnames_all = imagenet_classes
    #         classnames = []
    #         if set_id in ['A', 'R', 'V']:
    #             label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
    #             if set_id == 'R':
    #                 for i, m in enumerate(label_mask):
    #                     if m:
    #                         classnames.append(classnames_all[i])
    #             else:
    #                 classnames = [classnames_all[i] for i in label_mask]
    #         else:
    #             classnames = classnames_all
    #     if args.cocoop:
    #         model.prompt_generator.reset_classnames(classnames, args.arch)
    #         model = model.cpu()
    #         model_state = model.state_dict()
    #         model = model.cuda(args.gpu)
    #     else:
    #         model.reset_classnames(classnames, args.arch)

    val_data = Video_dataset(
    config.data.val_root, config.data.val_list, config.data.label_list,
    random_shift=False, num_segments=config.data.num_segments,
    modality=config.data.modality,
    image_tmpl=config.data.image_tmpl,
    test_mode=True,
    transform=torchvision.transforms.Compose([
        cropping,
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize(input_mean, input_std),
    ]),
    dense_sample=args.dense,
    test_clips=args.test_clips)

    # val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
    
    #(Pdb) len(val_dataset[0][0]) 64, each image val_dataset[0][0][0].shape is torch.Size([3, 224, 224])
    # logging.info("number of test samples: {}".format(len(val_dataset)))
    logging.info("number of test samples: {}".format(len(val_data)))
    # val_loader = torch.utils.data.DataLoader(
    #             val_dataset,
    #             batch_size=batchsize, shuffle=True,
    #             num_workers=args.workers, pin_memory=True)


    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    val_loader = DataLoader(val_data,
        batch_size=config.data.batch_size, num_workers=config.data.workers,
        sampler=val_sampler, pin_memory=True, drop_last=False)
    
    results[set_id] = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, config)

    # del val_dataset, val_loader
    try:
        logging.info("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
    except:
        logging.info("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))

    logging.info("======== Result Summary ========")
    logging.info("params: nstep	lr	bs")
    logging.info("params: {}	{}	{}".format(args.tta_steps, args.lr, args.batch_size))
    logging.info("\t\t [set_id] \t\t Top-1 acc. \t\t Top-5 acc.")
    for id in results.keys():
        logging.info("{}".format(id), end="	")
    logging.info("\n")
    for id in results.keys():
        logging.info("{:.2f}".format(results[id][0]), end="	")
    logging.info("\n")


def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, config):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()
    if not args.cocoop: # no need to reset cocoop because it's fixed
        with torch.no_grad():
            model.reset()
    end = time.time()

    for i, (images, target) in enumerate(val_loader):
        assert args.gpu is not None
        
        n_seg = config.data.num_segments #8 #1, 1152, 224, 224. 3x16x8 3 224 224
       
        images = images.view((-1, 3) + images.size()[-2:]) #3, 8, 3, 224, 224 #bxc, frame, c, h, w
        # vt, c, h, w = images.size() #viewxtime, c,h,w
        image_input = images.cuda(args.gpu, non_blocking=True)
        image = image_input.view((args.test_crops, -1, config.data.num_segments) + image_input.size()[-3:])[2][0] #middle crop, first full length
        target = target.cuda(args.gpu, non_blocking=True)
        # if isinstance(images, list):
        #     for k in range(len(images)):
        #         images[k] = images[k].cuda(args.gpu, non_blocking=True)
        #     image = images[0]
        # else:
        #     if len(images.size()) > 4:
        #         # when using ImageNet Sampler as the dataset
        #         assert images.size()[0] == 1
        #         images = images.squeeze(0)
        #     images = images.cuda(args.gpu, non_blocking=True)
        #     image = images
        # target = target.cuda(args.gpu, non_blocking=True)
        # if args.tpt:
        #     images = torch.cat(images, dim=0)

        # reset the tunable prompt to its initial state

        
        if not args.cocoop: # no need to reset cocoop because it's fixed
            if args.tta_steps > 0:
                with torch.no_grad():
                    model.reset()
            optimizer.load_state_dict(optim_state)
            test_time_tuning(model, image_input, optimizer, scaler, args, config)
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    image_feature, pgen_ctx = model.gen_ctx(images, args.tpt)
            optimizer = None
            pgen_ctx = test_time_tuning(model, (image_feature, pgen_ctx), optimizer, scaler, args)

        # The actual inference goes here
        if args.tpt:
            if args.cocoop:
                image_feature = image_feature[0].unsqueeze(0)
        
        #calibration
        # for name, param in model.named_parameters():
        #     if 'bias' in name:
        #         param.requires_grad = True  # Set bias parameters to trainable
        #     else:
        #         param.requires_grad = False  # Set weight parameters to not trainable

        # # Create a list of trainable parameters (bias parameters)
        # trainable_param = [param for param in model.parameters() if param.requires_grad]

        # Define the optimizer with trainable parameters (bias parameters)
        # calibration_optimizer = torch.optim.AdamW(trainable_param, 1e-4)
        # model.train()
        # random_noise = np.random.rand(*tuple(image.size()))
        # random_noise = torch.from_numpy(random_noise).cuda(args.gpu)
        # with torch.cuda.amp.autocast():
        #     p_cf = model(random_noise, config.data.num_segments)
        # p_cf_normalized = p_cf / p_cf.sum(dim=1, keepdim=True)
        # p_cf_equal = torch.ones(p_cf.shape)
        # p_cf_equal = p_cf_equal / p_cf_equal.sum(dim=1, keepdim=True)
        # p_cf_equal = p_cf_equal.cuda(args.gpu)
        # # epsilon = 1e-8
        # # p_cf_normalized += epsilon
        # # output_normalized += epsilon
        # kl_loss = torch.sum(p_cf_normalized * torch.log(p_cf_normalized / p_cf_equal), dim=1).mean()
        # calibration_optimizer.zero_grad()
        # # compute gradient and do SGD step
        # scaler.scale(kl_loss).backward()
        
        # # Unscales the gradients of optimizer's assigned params in-place
        # scaler.step(calibration_optimizer)
        # scaler.update()

        # print(model.state_dict()['text_encoder.ln_final.bias'])

        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if args.cocoop:
                    output = model((image_feature, pgen_ctx))
                else:
                    output = model(image, config.data.num_segments)
                    # p_cf = neutral_model(image, config.data.num_segments)

        
        #calibrate (diagonal_W)
        # W = np.linalg.inv(np.identity(config.data.num_classes) * p_cf.detach().cpu().numpy())
        # b = np.zeros([config.data.num_classes, 1])
        # label_probs = output[0].detach().cpu().numpy()
        # label_probs = label_probs / np.sum(label_probs) #normalize to 1
        # calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b
        # calibrate_label_probs = torch.from_numpy(calibrate_label_probs).cuda(args.gpu).reshape(1, -1)
        # cal1, cal5 = accuracy(calibrate_label_probs, target, topk=(1, 5))
        # measure accuracy and record loss

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # print(cal1, cal5, acc1, acc5)   
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        if i%10 == 0:
            logging.info('processing the {}th video'.format(i))
            logging.info('running accuracy top 1: {}'.format(top1.avg))
            logging.info('running accuracy top 5: {}'.format(top5.avg))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i)
    progress.display_summary()

    return [top1.avg, top5.avg]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--test_crops', type=int, default=3)
    parser.add_argument('--config', type=str, help='global config file')
    parser.add_argument('--dense', default=False, action="store_true",
                    help='use dense sample for test as in Non-local I3D')
    parser.add_argument('--test_clips', type=int, default=1) 
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--logging', type=str)
    main()