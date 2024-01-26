import logging
import os
import pickle
import sys

import numpy as np


curPath = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(curPath)
sys.path.append('../..')
print("当前的工作目录：",os.getcwd())
print("python搜索模块的路径集合",sys.path)
from model.modeling_kgcnbart import KGCNBartForConditionalGeneration
from data_preprocessing.TLDataset import TLDataset
from downstream.summarization.eval.metrics import avg_ir_metrics, bleu
from downstream.summarization.spt_args import add_args

import torch
from transformers import Seq2SeqTrainingArguments, SchedulerType, IntervalStrategy, EarlyStoppingCallback, BartConfig
import argparse
import enums
from typing import Union, Tuple

from data_preprocessing.pretrain.CodeDataset import CodeDataset
from data_preprocessing.pretrain.vocab import Vocab, load_vocab
# from model.configuration_bart import BartConfig
from model.general import human_format, count_params, layer_wise_parameters

from pretrain.callbacks import LogStateCallBack
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

def run_summarization(args,
              trained_model: Union[KGCNBartForConditionalGeneration, str] = None,
              trained_vocab: Union[Tuple[Vocab, Vocab, Vocab], str] = None):
    tasks = args.pre_train_tasks
    trained_model = args.trained_model
    trained_vocab = args.trained_vocab
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
        isinstance(trained_model, KGCNBartForConditionalGeneration), \
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
    train_dataset = TLDataset(
            args=args,
            logger=logger,
            entity_dict=entity_dict,
            dataset_type='train',
            language='java',
    )
    logger.info(f'The size of training set: {len(train_dataset)}')

    eval_dataset = TLDataset(
            args=args,
            logger=logger,
            entity_dict=entity_dict,
            dataset_type='valid',
            language='java',
    )
    logger.info(f'The size of eval set: {len(eval_dataset)}')

    test_dataset = TLDataset(
        args=args,
        logger=logger,
        entity_dict=entity_dict,
        dataset_type='test',
        language='java',
    )
    logger.info(f'The size of training set: {len(test_dataset)}')
    # if args.pre_train_subset_ratio:
    #     dataset = dataset.subset(args.pre_train_subset_ratio)
    #     logger.info(f'The pre-train dataset is trimmed to subset due to the argument: '
    #                 f'train_subset_ratio={args.pre_train_subset_ratio}')
    #     logger.info('The size of trimmed pre-train set: {}'.format(len(dataset)))
    # logger.info('Datasets loaded and parsed successfully')


    # --------------------------------------------------
    # vocabs
    # --------------------------------------------------
    logger.info('-' * 100)
    if trained_vocab:
        if isinstance(trained_vocab, tuple):
            logger.info('Vocabularies are passed through parameter')
            assert len(trained_vocab) == 2
            code_vocab, nl_vocab = trained_vocab
        else:
            logger.info('Loading vocabularies from files')
            code_vocab = load_vocab(vocab_root=trained_vocab, name=args.code_vocab_name)
            nl_vocab = load_vocab(vocab_root=trained_vocab, name=args.nl_vocab_name)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    logger.info('-' * 100)

    # Prepare model
    entity_embedding_path = os.path.join(args.kg_path, 'entity_embedding')
    relation_embedding_path = os.path.join(args.kg_path, 'relation_embedding')
    entity_embedding = np.array(pickle.load(open(entity_embedding_path, "rb")))
    entity_embedding = np.array(list(np.zeros((4, 768))) + list(entity_embedding))
    relation_embedding = np.array(pickle.load(open(relation_embedding_path, "rb")))


    if trained_model:
        if isinstance(trained_model, KGCNBartForConditionalGeneration):
            logger.info('Model is passed through parameter')
            model = trained_model
        else:
            logger.info('Loading the model from file')
            config = BartConfig.from_json_file(os.path.join(trained_model, 'config.json'))
            model = KGCNBartForConditionalGeneration.from_pretrained(os.path.join(trained_model), entity_weight=entity_embedding, relation_weight=relation_embedding)
    # log model statistic
    logger.info('Model trainable parameters: {}'.format(human_format(count_params(model))))
    table = layer_wise_parameters(model)
    logger.debug('Layer-wised trainable parameters:\n{}'.format(table))
    logger.info('Model built successfully')

    # --------------------------------------------------
    # trainer
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Initializing the running configurations')

    def decode_preds(preds):
        preds, labels = preds
        decoded_preds = nl_vocab.decode_batch(preds)
        decoded_labels = nl_vocab.decode_batch(labels)
        return decoded_labels, decoded_preds

    # compute metrics
    def compute_valid_metrics(eval_preds):
        decoded_labels, decoded_preds = decode_preds(eval_preds)
        refs = [ref.strip().split() for ref in decoded_labels]
        cans = [can.strip().split() for can in decoded_preds]
        result = {}
        result.update(bleu(references=refs, candidates=cans))
        return result

    def compute_test_metrics(eval_preds):
        decoded_labels, decoded_preds = decode_preds(eval_preds)
        result = {'references': decoded_labels, 'candidates': decoded_preds}
        refs = [ref.strip().split() for ref in decoded_labels]
        cans = [can.strip().split() for can in decoded_preds]
        result.update(bleu(references=refs, candidates=cans))
        # result.update(rouge_l(references=refs, candidates=cans))
        # result.update(avg_ir_metrics(references=refs, candidates=cans))
        # result.update(accuracy_for_sequence(references=refs, candidates=cans))
        return result

    training_args = Seq2SeqTrainingArguments(output_dir=os.path.join(args.checkpoint_root, enums.TASK_SUMMARIZATION),
                                             overwrite_output_dir=True,
                                             do_train=True,
                                             do_eval=True,
                                             do_predict=True,
                                             evaluation_strategy=IntervalStrategy.STEPS,
                                             eval_steps=2500,
                                             prediction_loss_only=False,
                                             auto_find_batch_size=True,
                                             # per_device_train_batch_size=1,
                                             # per_device_eval_batch_size=1,
                                             gradient_accumulation_steps=args.gradient_accumulation_steps,
                                             learning_rate=args.learning_rate,
                                             weight_decay=args.lr_decay_rate,
                                             max_grad_norm=args.grad_clipping_norm,
                                             num_train_epochs=args.n_epoch,
                                             lr_scheduler_type=SchedulerType.LINEAR,
                                             warmup_steps=args.warmup_steps,
                                             logging_dir=os.path.join(args.tensor_board_root, enums.TASK_SUMMARIZATION),
                                             logging_strategy=IntervalStrategy.STEPS,
                                             logging_steps=args.logging_steps,
                                             save_strategy=IntervalStrategy.STEPS,
                                             save_steps=2500,
                                             save_total_limit=3,
                                             seed=args.random_seed,
                                             fp16=args.fp16,
                                             dataloader_drop_last=False,
                                             run_name=args.model_name,
                                             load_best_model_at_end=True,
                                             metric_for_best_model='bleu',
                                             greater_is_better=True,
                                             ignore_data_skip=False,
                                             label_smoothing_factor=args.label_smoothing,
                                             report_to=['tensorboard'],
                                             dataloader_pin_memory=True,
                                             predict_with_generate=True)
    trainer = CodeTrainer(main_args=args,
                          task=enums.TASK_SUMMARIZATION,
                          code_vocab=code_vocab,
                          nl_vocab=nl_vocab,
                          model=model,
                          args=training_args,
                          data_collator=None,
                          train_dataset=train_dataset,
                          eval_dataset=eval_dataset,
                          tokenizer=nl_vocab,
                          model_init=None,
                          compute_metrics=compute_valid_metrics,
                          entity_dict=entity_dict,
                          callbacks=[
                              EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience),
                              LogStateCallBack()])
    logger.info('Running configurations initialized successfully')
    # --------------------------------------------------
    # train
    # --------------------------------------------------
    only_test = False
    if not only_test:
        logger.info('-' * 100)
        logger.info('Start training')
        # train_result = trainer.train()
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        logger.info('Training finished')
        trainer.save_model(args.model_root)
        trainer.save_state()
        metrics = train_result.metrics
        trainer.log_metrics(split='train', metrics=metrics)
        trainer.save_metrics(split='train', metrics=metrics)

        # --------------------------------------------------
        # eval
        # --------------------------------------------------
        # logger.info('-' * 100)
        # logger.info('Start evaluating')
        # eval_metrics = trainer.evaluate(metric_key_prefix='valid',
        #                                 max_length=args.max_decode_step,
        #                                 num_beams=args.beam_width)
        # trainer.log_metrics(split='valid', metrics=eval_metrics)
        # trainer.save_metrics(split='valid', metrics=eval_metrics)

    # --------------------------------------------------
    # predict
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Start testing')
    trainer.compute_metrics = compute_test_metrics
    predict_results = trainer.predict(test_dataset=test_dataset,
                                      metric_key_prefix='test',
                                      max_length=args.max_nl_len,
                                      num_beams=3)
    predict_metrics = predict_results.metrics
    references = predict_metrics.pop('test_references')
    candidates = predict_metrics.pop('test_candidates')
    trainer.log_metrics(split='test', metrics=predict_metrics)
    trainer.save_metrics(split='test', metrics=predict_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type', 'bool', lambda v: v.lower() in ['yes', 'true', 't', '1', 'y'])

    add_args(parser)
    main_args = parser.parse_args()
    run_summarization(main_args)