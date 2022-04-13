import os
import time
import json
import pickle
import itertools
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn
from torch.optim import LBFGS
from torch.distributions.multivariate_normal import MultivariateNormal

from FSLTask import FSLTaskMaker
from utils.io_utils import DataWriter, logger


def torch_logistic_reg_lbfgs_batch(X_aug, Y_aug, firth_c=0.0, max_iter=1000, verbose=True):
    batch_dim, n_samps, n_dim = X_aug.shape
    assert Y_aug.shape == (batch_dim, n_samps)
    num_classes = Y_aug.unique().numel()

    device = X_aug.device
    tch_dtype = X_aug.dtype
    # default value from https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html

    # from scipy.minimize.lbfgsb. In pytorch, it is the equivalent "max_iter"
    # (note that "max_iter" in torch.optim.LBFGS is defined per epoch and a step function call!)
    max_corr = 10
    tolerance_grad = 1e-05
    tolerance_change = 1e-09
    line_search_fn = 'strong_wolfe'
    l2_c = 1.0
    use_bias = True

    # According to https://github.com/scipy/scipy/blob/master/scipy/optimize/_lbfgsb_py.py#L339
    # wa (i.e., the equivalenet of history_size) is 2 * m * n (where m is max_corrections and n is the dimensions).
    history_size = max_corr * 2  # since wa is O(2*m*n) in size

    num_epochs = max_iter // max_corr  # number of optimization steps
    max_eval_per_epoch = None  # int(max_corr * max_evals / max_iter) matches the 15000 default limit in scipy!

    W = torch.nn.Parameter(torch.zeros((batch_dim, n_dim, num_classes), device=device, dtype=tch_dtype))
    opt_params = [W]
    linlayer = lambda x_: x_.matmul(W)
    if use_bias:
        bias = torch.nn.Parameter(torch.zeros((batch_dim, 1, num_classes), device=device, dtype=tch_dtype))
        opt_params.append(bias)
        linlayer = lambda x_: (x_.matmul(W) + bias)

    optimizer = LBFGS(opt_params, lr=1, max_iter=max_corr, max_eval=max_eval_per_epoch,
                      tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                      history_size=history_size, line_search_fn=line_search_fn)

    Y_aug_i64 = Y_aug.to(device=device, dtype=torch.int64)
    for epoch in range(num_epochs):
        if verbose:
            running_loss = 0.0

        inputs_, labels_ = X_aug, Y_aug_i64

        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()

            batch_dim_, n_samps_, n_dim_ = inputs_.shape
            outputs_ = linlayer(inputs_)
            # outputs_.shape -> batch_dim, n_samps, num_classes
            logp = outputs_ - torch.logsumexp(outputs_, dim=-1, keepdims=True)
            # logp.shape -> batch_dim, n_samps, num_classes
            label_logps = -logp.gather(dim=-1, index=labels_.reshape(batch_dim_, n_samps_, 1))
            # label_logps.shape -> batch_dim, n_samps, 1
            loss_cross = label_logps.mean(dim=(-1, -2)).sum(dim=0)
            loss_firth = -logp.mean(dim=(-1, -2)).sum(dim=0)
            loss_l2 = 0.5 * torch.square(W).sum() / n_samps_

            loss = loss_cross + firth_c * loss_firth + l2_c * loss_l2
            loss = loss / batch_dim_
            if loss.requires_grad:
                loss.backward()
            return loss

        # Update weights
        optimizer.step(closure)

        # Update the running loss
        if verbose:
            loss = closure()
            running_loss += loss.item()
            logger(f"Epoch: {epoch + 1:02}/{num_epochs} Loss: {running_loss:.5e}")
    return linlayer


def distribution_calibration_tch(query, base_means, base_cov, k, alpha=0.21):
    assert torch.is_tensor(query)
    assert torch.is_tensor(base_means)
    assert torch.is_tensor(base_cov)

    batch_dims, n_dim = query.shape[:-1], query.shape[-1]
    batch_dim = int(np.prod(batch_dims))

    n_classes = base_means.shape[0]
    assert base_means.shape == (n_classes, n_dim)
    assert base_cov.shape == (n_classes, n_dim, n_dim)

    base_means = base_means.unsqueeze(0).expand(batch_dim, n_classes, n_dim)
    base_cov = base_cov.unsqueeze(0).expand(batch_dim, n_classes, n_dim, n_dim)
    # query      -> shape = (batch_dim, n_dim)
    # base_means -> shape = (batch_dim, n_classes, n_dim)
    # base_cov   -> shape = (batch_dim, n_classes, n_dim, n_dim)

    dist = torch.linalg.norm(query.reshape(batch_dim, 1, n_dim) - base_means, 2,
                             dim=-1)  # dist.shape == (batch_dim, n_classes)
    index = torch.topk(dist, k, dim=-1, largest=False, sorted=True).indices  # index.shape == (batch_dim, k)

    gathered_mean = torch.gather(base_means, dim=-2, index=index.unsqueeze(-1).expand(batch_dim, k, n_dim))
    assert gathered_mean.shape == (batch_dim, k, n_dim)
    gathered_cov = torch.gather(base_cov, dim=-3,
                                index=index.unsqueeze(-1).unsqueeze(-1).expand(batch_dim, k, n_dim, n_dim))
    assert gathered_cov.shape == (batch_dim, k, n_dim, n_dim)

    mean = torch.cat([gathered_mean, query.reshape(batch_dim, 1, n_dim)],
                     dim=-2)  # mean.shape == (batch_dim, k+1, n_dim)

    calibrated_mean = torch.mean(mean, dim=-2)
    assert calibrated_mean.shape == (batch_dim, n_dim)
    calibrated_cov = torch.mean(gathered_cov, dim=-3) + alpha
    assert calibrated_cov.shape == (batch_dim, n_dim, n_dim)

    return calibrated_mean.reshape(*batch_dims, n_dim), calibrated_cov.reshape(*batch_dims, n_dim, n_dim)


def main(config_dict):
    config_id = config_dict['config_id']
    device_name = config_dict['device_name']
    rng_seed = config_dict['rng_seed']
    n_tasks = config_dict['n_tasks']
    source_dataset = config_dict['source_dataset']
    target_dataset = config_dict['target_dataset']
    n_shots_list = config_dict['n_shots_list']
    n_ways_list = config_dict['n_ways_list']
    split_name_list = config_dict['split_list']
    n_aug_list = config_dict['n_aug_list']
    firth_coeff_list = config_dict['firth_coeff_list']
    n_query = config_dict['n_query']
    dc_tukey_lambda = config_dict['dc_tukey_lambda']
    dc_k = config_dict['dc_k']
    dc_alpha = config_dict['dc_alpha']
    backbone_arch = config_dict['backbone_arch']
    backbone_method = config_dict['backbone_method']
    lbfgs_iters = config_dict['lbfgs_iters']
    store_results = config_dict['store_results']
    results_dir = config_dict['results_dir']
    features_dir = config_dict['features_dir']
    cache_dir = config_dict['cache_dir']
    dump_period = config_dict['dump_period']
    torch_threads = config_dict['torch_threads']
    task_bs = 10  # The number of tasks to stack to each other for parallel optimization

    dsname2abbrv = {'miniImagenet': 'mini', 'tieredImagenet': 'tiered', 'CUB': 'CUB'}

    data_writer = None
    if store_results:
        assert results_dir is not None, 'Please provide results_dir in the config_dict.'
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        data_writer = DataWriter(dump_period=dump_period)

    tch_dtype = torch.float32
    untouched_torch_thread = torch.get_num_threads()
    if torch_threads:
        torch.set_num_threads(torch_threads)

    for setting in itertools.product(firth_coeff_list, n_ways_list, n_shots_list, n_aug_list, split_name_list):
        firth_coeff, n_ways, n_shots, n_aug, split = setting
        os.makedirs(results_dir, exist_ok=True)
        np.random.seed(rng_seed+12345)
        torch.manual_seed(rng_seed+12345)

        src_ds_abbrv = dsname2abbrv.get(source_dataset, source_dataset)
        trg_ds_abbrv = dsname2abbrv.get(target_dataset, target_dataset)
        
        config_cols_dict = OrderedDict(n_shots=n_shots, n_ways=n_ways, source_dataset=source_dataset,
                                       target_dataset=target_dataset, backbone_arch=backbone_arch,
                                       backbone_method=backbone_method, n_aug=n_aug, split=split,
                                       firth_coeff=firth_coeff, n_query=n_query,
                                       dc_tukey_lambda=dc_tukey_lambda, dc_k=dc_k,
                                       dc_alpha=dc_alpha, lbfgs_iters=lbfgs_iters,
                                       rng_seed=rng_seed)
        print('-'*80)
        logger('Current configuration:')
        for (cfg_key_, cfg_val_) in config_cols_dict.items():
            logger(f"  --> {cfg_key_}: {cfg_val_}", flush=True)

        task_maker = FSLTaskMaker()
        task_maker.reset_global_vars()

        features_bb_dir = f"{features_dir}/{backbone_arch}_{backbone_method}"
        Path(features_bb_dir).mkdir(parents=True, exist_ok=True)
        task_maker.loadDataSet(f'{src_ds_abbrv}2{trg_ds_abbrv}_{split}', features_dir=features_bb_dir)
        logger("* Target Dataset loaded", flush=True)

        n_lsamples = n_ways * n_shots
        n_usamples = n_ways * n_query
        n_samples = n_lsamples + n_usamples

        cfg = {'n_shots': n_shots, 'n_ways': n_ways, 'n_query': n_query, 'seed': rng_seed}
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        task_maker.setRandomStates(cfg, cache_dir=cache_dir)
        ndatas = task_maker.GenerateRunSet(end=n_tasks, cfg=cfg)
        ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_tasks, n_samples, -1)
        labels = torch.arange(n_ways).view(1, 1, n_ways)
        labels = labels.expand(n_tasks, n_shots + n_query, n_ways)
        labels = labels.clone().view(n_tasks, n_samples)

        # ---- Base class statistics
        base_means = []
        base_cov = []
        base_features_path = f"{features_dir}/{backbone_arch}_{backbone_method}/{src_ds_abbrv}2{src_ds_abbrv}_base.pkl"
        logger(f"* Reading Base Features from {base_features_path}", flush=True)
        with open(base_features_path, 'rb') as fp:
            data = pickle.load(fp)
            for key in data.keys():
                feature = np.array(data[key])
                mean = np.mean(feature, axis=0)
                cov = np.cov(feature.T)
                base_means.append(mean)
                base_cov.append(cov)

        logger("* Means and Covariance Matrices calculated", flush=True)
        with torch.no_grad():
            base_means_tch = torch.cat(
                [torch.from_numpy(x).unsqueeze(0).to(device=device_name, dtype=tch_dtype) for x in base_means])
            base_cov_tch = torch.cat([torch.from_numpy(x).unsqueeze(0).to(device=device_name, dtype=tch_dtype)
                                      for x in base_cov])
        # ---- classification for each task
        test_acc_list = []
        logger(f'* Starting Classification for {n_tasks} Tasks...')
        st_time = time.time()

        time_lst_calib = []
        time_lst_gen = []
        time_lst_lbfgs = []

        all_run_idxs = np.arange(n_tasks)
        all_run_idxs = all_run_idxs.reshape(-1, task_bs)

        n_dim = ndatas.shape[-1]
        for ii, run_idxs in enumerate(all_run_idxs):
            run_idxs = run_idxs.astype(int).tolist()
            batch_dim = len(run_idxs)

            support_data = ndatas[run_idxs][:, :n_lsamples, :].to(device=device_name, dtype=tch_dtype)
            assert support_data.shape == (batch_dim, n_lsamples, n_dim)

            support_label = labels[run_idxs][:, :n_lsamples].to(device=device_name, dtype=torch.int64)
            assert support_label.shape == (batch_dim, n_lsamples)

            query_data = ndatas[run_idxs][:, n_lsamples:, :].to(device=device_name, dtype=tch_dtype)
            assert query_data.shape == (batch_dim, n_usamples, n_dim)

            query_label = labels[run_idxs][:, n_lsamples:].to(device=device_name, dtype=torch.int64)
            assert query_label.shape == (batch_dim, n_usamples)
            # ---- Tukey's transform

            support_data = torch.pow(support_data, dc_tukey_lambda)
            query_data = torch.pow(query_data, dc_tukey_lambda)

            # ---- distribution calibration and feature sampling
            num_sampled = int(n_aug / n_shots)

            if num_sampled == 0:
                X_aug, Y_aug = support_data, support_label
            else:
                start_time = time.time()
                with torch.no_grad():
                    mean_tch, cov_tch = distribution_calibration_tch(support_data, base_means_tch, base_cov_tch,
                                                                     alpha=dc_alpha, k=dc_k)
                assert mean_tch.shape == (batch_dim, n_lsamples, n_dim)
                assert cov_tch.shape == (batch_dim, n_lsamples, n_dim, n_dim)
                time_lst_calib.append(time.time() - start_time)

                start_time = time.time()
                samps_at_a_time = 1
                with torch.no_grad():
                    sampled_data_lst = []
                    mvn_gen = MultivariateNormal(mean_tch, covariance_matrix=cov_tch)
                    for _ in range(int(np.ceil(float(num_sampled) / samps_at_a_time))):
                        norm_samps_tch = mvn_gen.sample((samps_at_a_time,))
                        # norm_samps_tch.shape -> (samps_at_a_time, batch_dim, n_lsamples, n_dim)
                        sampled_data_lst.append(norm_samps_tch)
                    sampled_data = torch.cat(sampled_data_lst, dim=0)[:num_sampled]
                    # sampled_data.shape -> (num_sampled, batch_dim, n_lsamples, n_dim)
                    assert sampled_data.shape == (num_sampled, batch_dim, n_lsamples, n_dim)
                    sampled_data = sampled_data.permute(1, 2, 0, 3)
                    assert sampled_data.shape == (batch_dim, n_lsamples, num_sampled, n_dim)
                    time_lst_gen.append(time.time() - start_time)

                with torch.no_grad():
                    sampled_label__ = support_label.unsqueeze(-1)
                    sampled_label_ = sampled_label__.expand(batch_dim, n_lsamples, num_sampled)
                    sampled_label = sampled_label_.reshape(batch_dim, n_lsamples * num_sampled)
                    sampled_data = sampled_data.reshape(batch_dim, n_lsamples * num_sampled, n_dim)

                    X_aug = torch.cat([support_data, sampled_data], dim=-2)
                    # X_aug.shape -> batch_dim, n_lsamples + n_lsamples* num_sampled, n_dim
                    Y_aug = torch.cat([support_label, sampled_label], dim=-1)
                    # Y_aug.shape -> batch_dim, n_lsamples + n_lsamples*num_sampled
            # ---- train classifier

            start_time = time.time()
            classifier = torch_logistic_reg_lbfgs_batch(X_aug, Y_aug, firth_coeff,
                                                        max_iter=lbfgs_iters, verbose=False)
            with torch.no_grad():
                predicts = classifier(query_data).argmax(dim=-1)
                # predicts.shape -> batch_dim, n_usamples
            time_lst_lbfgs.append(time.time() - start_time)

            with torch.no_grad():
                acc = (predicts == query_label).double().mean(dim=(-1)).detach().cpu().numpy().ravel()
            test_acc_list += acc.tolist()

            runs_so_far = len(test_acc_list)

            if (ii + 1) % 2 == 0:
                time_per_iter = (time.time() - st_time) / runs_so_far
                acc_mean = 100 * np.mean(test_acc_list)
                acc_ci = 1.96 * 100.0 * float(np.std(test_acc_list) / np.sqrt(len(test_acc_list)))
                print('.' * acc.size + f' (Accuracy So Far: {acc_mean:.2f} +/- {acc_ci:.2f},    ' + 
                      f'{time_per_iter:.3f} sec/iter,    {runs_so_far:05d}/{n_tasks:05d} Tasks Done)', 
                      flush=True)
            else:
                logger('.' * acc.size, end='', flush=True)

        tam = 100.0 * float(np.mean(test_acc_list))
        tac = 1.96 * 100.0 * float(np.std(test_acc_list) / np.sqrt(len(test_acc_list)))
        logger(f' --> Final Accuracy: {tam:.2f} +/- {tac:.2f}' + '%', flush=True)

        if store_results:
            csv_path = f'{results_dir}/{config_id}.csv'
            for task_id, task_acc in enumerate(test_acc_list):
                row_dict = config_cols_dict.copy()  # shallow copy
                row_dict['task_id'] = task_id
                row_dict['test_acc'] = task_acc
                data_writer.add(row_dict, csv_path)

    if store_results:
        # We need to make a final dump before exiting to make sure all data is stored
        data_writer.dump()

    torch.set_num_threads(untouched_torch_thread)


if __name__ == '__main__':
    use_argparse = True
    if use_argparse:
        import argparse
        my_parser = argparse.ArgumentParser()
        my_parser.add_argument('--configid', action='store', type=str, required=True)
        my_parser.add_argument('--device', action='store', type=str, required=True)
        args = my_parser.parse_args()
        args_configid = args.configid
        args_device_name = args.device
    else:
        args_configid = '1_mini2CUB/1s10w_0aug'
        args_device_name = 'cuda:0'

    if '/' in args_configid:
        args_configid_split = args_configid.split('/')
        my_config_id = args_configid_split[-1]
        config_tree = '/'.join(args_configid_split[:-1])
    else:
        my_config_id = args_configid
        config_tree = ''

    PROJPATH = os.getcwd()
    cfg_dir = f'{PROJPATH}/configs'
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = f'{PROJPATH}/configs/{config_tree}/{my_config_id}.json'
    logger(f'Reading Configuration from {cfg_path}', flush=True)

    with open(cfg_path) as f:
        proced_config_dict = json.load(f)

    proced_config_dict['config_id'] = my_config_id
    proced_config_dict['device_name'] = args_device_name
    proced_config_dict['results_dir'] = f'{PROJPATH}/results/{config_tree}'
    proced_config_dict['cache_dir'] = f'{PROJPATH}/cache'
    proced_config_dict['features_dir'] = f'{PROJPATH}/features'

    main(proced_config_dict)
