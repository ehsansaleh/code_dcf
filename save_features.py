import os
import pickle
import glob
import collections
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from data.datamgr import SimpleDataManager
import wrn_model


class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != 'best.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):    
    best_file = os.path.join(checkpoint_dir, 'best.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)


def extract_feature(val_loader, model, save_dir, split='base', device=None):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    if os.path.isfile(save_dir + '/%s.pkl' % split):
        data = load_pickle(save_dir + '/%s.pkl' % split)
        return data
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    # model.eval()
    with torch.no_grad():
        output_dict = collections.defaultdict(list)
        print(f'length of val loader is {len(val_loader)}')
        st_time = time.time()
        for i, (inputs, labels) in enumerate(val_loader):
            # compute output
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, _ = model(inputs)
            outputs = outputs.cpu().data.numpy()
            
            for out, label in zip(outputs, labels):
                output_dict[label.item()].append(out)
            if (i+1) % 10 == 0:
                print(f'. (%.3f sec/iter)' % ((time.time() - st_time)/(i+1)), flush=True)
            else:
                print('.', end='', flush=True)
    
        all_info = output_dict
        feature_file = save_dir + '/%s.pkl' % split
        print(f'Saving features to {feature_file}')
        save_pickle(feature_file, all_info)
        return all_info


if __name__ == '__main__':
    use_argparse = True
    if use_argparse:
        import argparse
        parser = argparse.ArgumentParser(description='Feature generation and saving script')
        parser.add_argument('--source_dataset', default='miniImagenet',        
                            help='dataset used for backbone training e.g. miniImagenet/tieredImagenet')
        parser.add_argument('--target_dataset', default='CUB',
                            help='the dataset for which the features are ' +
                                 'extracted e.g. miniImagenet/tieredImagenet/CUB')
        parser.add_argument('--model', default='WideResNet28_10',      
                            help='model:  WideResNet28_10 /Conv{4|6} /ResNet{10|18|34|50|101}')
        parser.add_argument('--method', default='S2M2_R', help='rotation/manifold_mixup/S2M2_R')
        parser.add_argument('--split', default='novel', help='base/val/novel')
        parser.add_argument('--device', default='cuda:0', help='cuda/cuda:0/cuda:1/cpu')
        args = parser.parse_args()
        args_source_dataset = args.source_dataset
        args_target_dataset = args.target_dataset
        args_model = args.model
        args_method = args.method
        args_split = args.split
        args_device_name = args.device
    else:
        args_source_dataset = 'miniImagenet'
        args_target_dataset = 'CUB'
        args_model = 'WideResNet28_10'
        args_method = 'S2M2_R'
        args_split = 'novel'
        args_device_name = "cuda:0"
    
    device = torch.device(args_device_name)
    msg_ = f"the provided split {args_split} is not valid!"
    assert (args_split in ['base', 'novel', 'val']), msg_
    print(f'* Extracting features for {args_split} set of {args_target_dataset} ' + 
          f'using the backbone trained on {args_source_dataset}')
    
    save_dir = './features'
    filelists_dir = './filelists'
    checkpoints_dir = './checkpoints'
    
    loadfile = f'{filelists_dir}/{args_target_dataset}/{args_split}.json'
    print(f'* Reading filelists from {loadfile}')
    assert os.path.exists(loadfile), (f'{loadfile} does not exist. Please download/link the '
                                      f'datasets and then run the json_make script in the `filelists` '
                                      f'directory to create the json filelists.')
    if args_target_dataset == 'miniImagenet' or args_target_dataset == 'CUB':
        datamgr = SimpleDataManager(84, batch_size=256)
    elif args_target_dataset == 'tieredImagenet':
        datamgr = SimpleDataManager(84, batch_size=256)
    else:
        raise ValueError(f'Pre-processing for {args_target_dataset} is not implemented.')
    loader = datamgr.get_data_loader(loadfile, aug=False)

    backbone_model_dir = f'{checkpoints_dir}/{args_source_dataset}/{args_model}_{args_method}'
    print(f"* Backbone model directory: {backbone_model_dir}")
    
    modelfile = get_resume_file(backbone_model_dir)
    print(f"* Reading backbone model from {modelfile}")
    if args_model == 'WideResNet28_10':
        model = wrn_model.wrn28_10(num_classes=200)  # num_classes set to default 200
    else:
        raise ValueError(f'Backbone instantiation for {args_model} not implemented.')
        
    if 'cuda' in args_device_name:
        model = model.to(device)
        cudnn.benchmark = True
        checkpoint = torch.load(modelfile)
    else:
        checkpoint = torch.load(modelfile, map_location=args_device_name)

    state = checkpoint['state']
    state_keys = list(state.keys())

    callwrap = False
    if 'module' in state_keys[0]:
        callwrap = True
    if callwrap:
        model = WrappedModel(model)
    model_dict_load = model.state_dict()
    model_dict_load.update(state)
    model.load_state_dict(model_dict_load, strict=False)
    model.eval()
        
    features_savedir = f'{save_dir}/{args_model}_{args_method}'
    dsname2abbrv = {'miniImagenet': 'mini', 'tieredImagenet': 'tiered', 'CUB': 'CUB'}
    src_ds_abbrv = dsname2abbrv.get(args_source_dataset, args_source_dataset)
    trg_ds_abbrv = dsname2abbrv.get(args_target_dataset, args_target_dataset)
    fname = f'{src_ds_abbrv}2{trg_ds_abbrv}_{args_split}'  # Example: fname = 'mini2CUB_novel'
    
    output_dict = extract_feature(loader, model, features_savedir, split=fname, device=device)
