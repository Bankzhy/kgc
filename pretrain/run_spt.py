import logging
import os
import sys
curPath = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(curPath)
print("当前的工作目录：",os.getcwd())
print("python搜索模块的路径集合",sys.path)
import torch
from transformers import Seq2SeqTrainingArguments, SchedulerType, IntervalStrategy, BartForConditionalGeneration, \
    BartConfig
import argparse
import enums
from typing import Union, Tuple

from data_preprocessing.pretrain.CodeDataset import CodeDataset
from data_preprocessing.pretrain.vocab import Vocab
# from model.configuration_bart import BartConfig
from model.general import human_format, count_params, layer_wise_parameters
# from model.modeling_bart import BartForConditionalGeneration
from pretrain.callbacks import LogStateCallBack
from pretrain.spt_args import add_args
from pretrain.trainer import CodeTrainer

logger = logging.getLogger(__name__)
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

def pre_train(args,
              trained_model: Union[BartForConditionalGeneration, str] = None,
              trained_vocab: Union[Tuple[Vocab, Vocab, Vocab], str] = None):
    tasks = args.pre_train_tasks
    if tasks is None:
        logger.warning('Was specified for pre-training, but got pre-training tasks to None, '
                       'will default to {}'.format(','.join(enums.PRE_TRAIN_TASKS)))
        tasks = enums.PRE_TRAIN_TASKS
    else:
        supported_tasks = []
        for task in tasks.split(','):
            task = task.strip().lower()
            if task in enums.PRE_TRAIN_TASKS:
                supported_tasks.append(task)
            else:
                logger.warning(f'Pre-training task {task} is not supported and will be ignored.')
        tasks = supported_tasks

    assert not trained_model or \
        isinstance(trained_model, str) or \
        isinstance(trained_model, BartForConditionalGeneration), \
        f'The model type is not supported, expect Bart model or string of model dir, got {type(trained_model)}'

    if trained_vocab is None and args.trained_vocab is not None:
        trained_vocab = args.trained_vocab
    assert not trained_vocab or isinstance(trained_vocab, str), \
        f'The vocab type is not supported, expect string of vocab dir, got {type(trained_vocab)}'

    logger.info('*' * 100)
    logger.info('Initializing pre-training environments')

    # --------------------------------------------------
    # datasets
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Loading and parsing datasets')
    entity_dict = load_entity_dict(args)
    dataset = CodeDataset(
            args=args,
            logger=logger,
            entity_dict=entity_dict,
            dataset_type='train',
            task=enums.TASK_NSP,
            language='java',
    )
    logger.info(f'The size of pre_training set: {len(dataset)}')
    if args.pre_train_subset_ratio:
        dataset = dataset.subset(args.pre_train_subset_ratio)
        logger.info(f'The pre-train dataset is trimmed to subset due to the argument: '
                    f'train_subset_ratio={args.pre_train_subset_ratio}')
        logger.info('The size of trimmed pre-train set: {}'.format(len(dataset)))
    logger.info('Datasets loaded and parsed successfully')

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Building model')
    config = BartConfig(vocab_size=len(dataset.code_tokenizer) + len(dataset.nl_tokenizer),
                        max_position_embeddings=512,
                        encoder_layers=args.n_layer,
                        encoder_ffn_dim=args.d_ff,
                        encoder_attention_heads=args.n_head,
                        decoder_layers=args.n_layer,
                        decoder_ffn_dim=args.d_ff,
                        decoder_attention_heads=args.n_head,
                        activation_function='gelu',
                        d_model=args.d_model,
                        dropout=args.dropout,
                        use_cache=True,
                        pad_token_id=Vocab.START_VOCAB.index(Vocab.PAD_TOKEN),
                        bos_token_id=Vocab.START_VOCAB.index(Vocab.SOS_TOKEN),
                        eos_token_id=Vocab.START_VOCAB.index(Vocab.EOS_TOKEN),
                        is_encoder_decoder=True,
                        decoder_start_token_id=Vocab.START_VOCAB.index(Vocab.SOS_TOKEN),
                        forced_eos_token_id=Vocab.START_VOCAB.index(Vocab.EOS_TOKEN),
                        max_length=100,
                        min_length=1,
                        num_beams=args.beam_width,
                        num_labels=2)
    model = BartForConditionalGeneration(config=config)
    # log model statistic
    logger.info('Model trainable parameters: {}'.format(human_format(count_params(model))))
    table = layer_wise_parameters(model)
    logger.debug('Layer-wised trainable parameters:\n{}'.format(table))
    logger.info('Model built successfully')

    # --------------------------------------------------
    # pre-train
    # --------------------------------------------------
    for task in tasks:
        logger.info('-' * 100)
        logger.info(f'Pre-training task: {task.upper()}')

        # if isinstance(dataset, torch.utils.data.Subset):
        #     dataset.dataset.set_task(task)
        # else:
        #     dataset.set_task(task)
        # set model mode
        logger.info('-' * 100)
        # model.set_model_mode(enums.MODEL_MODE_GEN)
        # --------------------------------------------------
        # trainer
        # --------------------------------------------------
        logger.info('-' * 100)
        logger.info('Initializing the running configurations')
        training_args = Seq2SeqTrainingArguments(output_dir=os.path.join(args.pre_train_output_root, task),
                                                 overwrite_output_dir=True,
                                                 do_train=True,
                                                 per_device_train_batch_size=args.batch_size,
                                                 gradient_accumulation_steps=1,
                                                 learning_rate=args.learning_rate,
                                                 weight_decay=args.lr_decay_rate,
                                                 max_grad_norm=args.grad_clipping_norm,
                                                 num_train_epochs=5,
                                                 lr_scheduler_type=SchedulerType.LINEAR,
                                                 warmup_steps=args.warmup_steps,
                                                 logging_dir=os.path.join(args.tensor_board_root, task),
                                                 logging_strategy=IntervalStrategy.STEPS,
                                                 logging_steps=args.logging_steps,
                                                 save_strategy=IntervalStrategy.NO,
                                                 seed=args.random_seed,
                                                 fp16=args.fp16,
                                                 dataloader_drop_last=False,
                                                 run_name=args.model_name,
                                                 load_best_model_at_end=True,
                                                 ignore_data_skip=False,
                                                 label_smoothing_factor=args.label_smoothing,
                                                 report_to=['tensorboard'],
                                                 dataloader_pin_memory=True)
        trainer = CodeTrainer(main_args=args,
                              task=task,
                              model=model,
                              args=training_args,
                              data_collator=None,
                              train_dataset=dataset,
                              model_init=None,
                              compute_metrics=None,
                              callbacks=[LogStateCallBack()])
        logger.info('Running configurations initialized successfully')

        # --------------------------------------------------
        # train
        # --------------------------------------------------
        logger.info('-' * 100)
        logger.info(f'Start pre-training task: {task}')
        # model device
        logger.info('Device: {}'.format(next(model.parameters()).device))
        mass_result = trainer.train()
        logger.info(f'Pre-training task {task} finished')
        trainer.save_model(os.path.join(args.model_root, task))


def run():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type', 'bool', lambda v: v.lower() in ['yes', 'true', 't', '1', 'y'])

    add_args(parser)
    main_args = parser.parse_args()
    pre_train(main_args)

if __name__ == '__main__':
    run()