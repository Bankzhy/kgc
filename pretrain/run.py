import sys
import os

from torch import nn

curPath = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(curPath)
print("当前的工作目录：",os.getcwd())
print("python搜索模块的路径集合",sys.path)
import argparse
import json
import math
import os
import logging
import pickle
from pathlib import Path

import torch.distributed as dist
import torch
import random
import numpy as np
import glob
from torch.utils.data import RandomSampler, DistributedSampler
from tqdm import tqdm

import enums
from data_preprocessing.pretrain.CodeDataset import CodeDataset
from data_preprocessing.pretrain.vocab import load_vocab, init_vocab
from model import KGBartForConditionalGeneration
from model.configuration_bart import BartConfig
from nn.data_parallel import DataParallelImbalance
from pretrain.pretrain_args import add_pretrain_args
from model.optimization import AdamW, get_linear_schedule_with_warmup

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        if x[0] is None:
            batch_tensors.append(None)
        elif isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors

def _get_max_epoch_model(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "model.*.bin"))
    fn_optim_list = glob.glob(os.path.join(output_dir, "optim.*.bin"))
    fn_sched_list = glob.glob(os.path.join(output_dir, "sched.*.bin"))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    both_set = set([int(Path(fn).stem.split('.')[-1]) for fn in fn_model_list]
                   ) & set([int(Path(fn).stem.split('.')[-1]) for fn in fn_optim_list]) & set(
        [int(Path(fn).stem.split('.')[-1]) for fn in fn_sched_list])
    if both_set:
        return max(both_set)
    else:
        return None

def trange(*args, **kwargs):
    """Shortcut for tqdm(range(*args), **kwargs)."""
    return tqdm(range(*args), **kwargs)

def load_entity_dict(args):
    dict_path = os.path.join(args.kg_path, 'entity_dict.txt')
    dict = {}
    with open(dict_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            ll = line.split(";")
            idx = ll[1].replace("\n", "")
            idx = int(idx)
            dict[ll[0]] = idx
    return dict

def run():
    parser = argparse.ArgumentParser()
    add_pretrain_args(parser)

    args = parser.parse_args()

    # assert Path(args.model_recover_path).exists(
    # ), "--model_recover_path doesn't exist"

    args.output_dir = args.output_dir.replace(
        '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))
    args.log_dir = args.log_dir.replace(
        '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))

    entity_dict = load_entity_dict(args)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'opt.json'), 'w'), sort_keys=True, indent=2)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        dist.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(
        args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    if args.local_rank not in (-1, 0):
        # Make sure only the first process in distributed training will download model & vocab
        dist.barrier()

    if args.do_train:
        print("Loading Train Dataset", args.dataset_dir)
        train_dataset = CodeDataset(
            args=args,
            logger=logger,
            task=enums.TASK_MASS,
            language='java',
            entity_dict=entity_dict
        )
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset, replacement=False)
            _batch_size = args.train_batch_size
        else:
            train_sampler = DistributedSampler(train_dataset)
            _batch_size = args.train_batch_size // dist.get_world_size()

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, sampler=train_sampler,
                                                       num_workers=args.num_workers,
                                                       collate_fn=batch_list_to_batch_tensors,
                                                       pin_memory=False)
    t_total = int(len(train_dataloader) * args.num_train_epochs /
                  args.gradient_accumulation_steps)
    # Prepare model
    entity_embedding_path = os.path.join(args.kg_path, 'entity_embedding')
    relation_embedding_path = os.path.join(args.kg_path, 'relation_embedding')
    entity_embedding = np.array(pickle.load(open(entity_embedding_path, "rb")))
    entity_embedding = np.array(list(np.zeros((4, 1024))) + list(entity_embedding))
    relation_embedding = np.array(pickle.load(open(relation_embedding_path, "rb")))

    recover_step = _get_max_epoch_model(args.output_dir)
    # cls_num_labels = 2
    # type_vocab_size = 6 + \
    #                   (1 if args.s2s_add_segment else 0) if args.new_segment_ids else 2
    # num_sentlvl_labels = 2 if args.pretraining_KG else 0
    # relax_projection = 4 if args.relax_projection else 0
    if args.local_rank not in (-1, 0):
        # Make sure only the first process in distributed training will download model & vocab
        dist.barrier()
    if (recover_step is None) and (args.model_recover_path is None):
        # if _state_dict == {}, the parameters are randomly initialized
        # if _state_dict == None, the parameters are initialized with bert-init
        _state_dict = {} if args.from_scratch else None
        model = KGBartForConditionalGeneration(BartConfig(
            vocab_size=train_dataset.get_vocab_size()
        ), entity_weight=entity_embedding,
                                                               relation_weight=relation_embedding)
        global_step = 0
    else:
        if recover_step:
            logger.info("***** Recover model: %d *****", recover_step)
            model_recover = torch.load(os.path.join(
                args.output_dir, "model.{0}.bin".format(recover_step)), map_location='cpu')
            # recover_step == number of epochs
            global_step = math.floor(
                recover_step * t_total / args.num_train_epochs)
        elif args.model_recover_path:
            logger.info("***** Recover model: %s *****",
                        args.model_recover_path)
            model_recover = torch.load(
                args.model_recover_path, map_location='cpu')
            global_step = 0
        model = KGBartForConditionalGeneration.from_pretrained(args.model_recover_path, state_dict=model_recover,
                                                               entity_weight=entity_embedding,
                                                               relation_weight=relation_embedding)


    if args.local_rank == 0:
        dist.barrier()

    # model.to(device)
    # if args.local_rank != -1:
    #     try:
    #         from torch.nn.parallel import DistributedDataParallel as DDP
    #     except ImportError:
    #         raise ImportError("DistributedDataParallel")
    #     model = DDP(model, device_ids=[
    #         args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    # elif n_gpu > 1:
    #     # model = torch.nn.DataParallel(model)
    #     model = DataParallelImbalance(model)

    model.to(device)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if recover_step:
        logger.info("***** Recover optimizer: %d *****", recover_step)
        optim_recover = torch.load(os.path.join(
            args.output_dir, "optim.{0}.bin".format(recover_step)), map_location='cpu')
        if hasattr(optim_recover, 'state_dict'):
            optim_recover = optim_recover.state_dict()
        optimizer.load_state_dict(optim_recover)
        schedule_recover = torch.load(os.path.join(
            args.output_dir, "sched.{0}.bin".format(recover_step)), map_location='cpu')
        scheduler.load_state_dict(schedule_recover)

        if args.loss_scale == 0:
            logger.info("***** Recover optimizer: dynamic_loss_scale *****")
            optimizer.dynamic_loss_scale = True

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.local_rank != -1:
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("DistributedDataParallel")
        model = DDP(model, device_ids=[
            args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    elif n_gpu > 1:
        # model = torch.nn.DataParallel(model)
        model = DataParallelImbalance(model)

    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    best_dev_loss = 1000

    output_eval_file = os.path.join(args.log_dir, "eval_results.txt")
    writer = open(output_eval_file, "w")
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", t_total)

        if recover_step:
            start_epoch = recover_step + 1
        else:
            start_epoch = 1
        for i_epoch in trange(start_epoch, int(args.num_train_epochs) + 1, desc="Epoch",
                              disable=args.local_rank not in (-1, 0)):
            model.train()
            if args.local_rank != -1:
                train_sampler.set_epoch(i_epoch)
            # iter_bar = tqdm(BackgroundGenerator(train_dataloader), desc='Iter (loss=X.XXX)',
            #                 disable=args.local_rank not in (-1, 0))
            for step, batch in enumerate(tqdm(train_dataloader, desc="Training", position=0, leave=True)):
                batch = [
                    t.to(device) if t is not None else None for t in batch]
                # if args.pretraining_KG:
                #     input_ids, input_entity_ids, subword_mask, word_mask, word_subword, decoder_input_ids, decoder_attention_mask, labels = batch
                # else:
                #     input_ids, input_entity_ids, subword_mask, word_mask, word_subword, decoder_input_ids, decoder_attention_mask, labels = batch

                input_ids, encoder_attention_mask, input_entity_ids, word_mask, decoder_input_ids, decoder_attention_mask, labels = batch

                loss_output = model(input_ids, input_entity_ids=input_entity_ids, attention_mask=encoder_attention_mask,
                                    word_mask=word_mask,

                                    decoder_input_ids=decoder_input_ids,
                                    decoder_attention_mask=decoder_attention_mask, labels=labels,
                                    label_smoothing=False)

                masked_lm_loss = loss_output.loss
                if n_gpu > 1:  # mean() to average on multi-gpu.
                    # loss = loss.mean()
                    masked_lm_loss = masked_lm_loss.mean()
                loss = masked_lm_loss

                # logging for each step (i.e., before normalization by args.gradient_accumulation_steps)
                # iter_bar.set_description('Iter %d (loss=%5.3f)' % (i_epoch, loss.item()))

                if step % 1000 == 0:
                    print('Iter %d  (Gen_loss=%5.3f)' % (i_epoch, loss.item()))

                # ensure that accumlated gradients are normalized
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

if __name__ == '__main__':
    run()