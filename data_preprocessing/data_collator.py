import re

import torch

from typing import List
import itertools

from .pretrain.vocab import Vocab
import enums


def collate_fn(batch, args, task, entity_dict, code_vocab, nl_vocab, ast_vocab=None):
    """
    Data collator function.

    Args:
        batch (list):
        args (argparse.Namespace):
        task (str):
        code_vocab (Vocab):
        nl_vocab (Vocab):
        ast_vocab (Vocab):

    Returns:
        dict: Model inputs

    """
    model_inputs = {}

    # mass
    if task == enums.TASK_MASS:

        code_raw, target_raw = map(list, zip(*batch))

        model_inputs['input_ids'], model_inputs['attention_mask'] = get_concat_batch_inputs(
            code_raw=code_raw,
            code_vocab=code_vocab,
            max_code_len=args.max_code_len,
            ast_raw=None,
            ast_vocab=ast_vocab,
            max_ast_len=0,
            nl_raw=None,
            nl_vocab=nl_vocab,
            max_nl_len=args.max_nl_len,
            no_ast=True,
            no_nl=True
        )

        # model_inputs['input_entity_ids'] = get_entity_ids(input_tokens=code_raw.split(), entity_dict=entity_dict)
        model_inputs['input_entity_ids'], model_inputs['word_mask'] = get_batch_entity_ids(args=args, input_tokens=code_raw, entity_dict=entity_dict)
        model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_batch_inputs(
            batch=target_raw,
            vocab=code_vocab,
            processor=Vocab.sos_processor,
            max_len=int(args.mass_mask_ratio * args.max_code_len)
        )
        model_inputs['labels'], _ = get_batch_inputs(batch=target_raw,
                                                     vocab=code_vocab,
                                                     processor=Vocab.eos_processor,
                                                     max_len=int(args.mass_mask_ratio * args.max_code_len))
    # nsp
    elif task == enums.TASK_NSP:

        code_raw, nl_raw = map(list, zip(*batch))

        model_inputs['input_ids'], model_inputs['attention_mask'] = get_concat_batch_inputs(
            code_raw=code_raw,
            code_vocab=code_vocab,
            max_code_len=args.max_code_len,
            ast_raw=None,
            ast_vocab=ast_vocab,
            max_ast_len=0,
            nl_raw=None,
            nl_vocab=nl_vocab,
            max_nl_len=args.max_nl_len,
            no_ast=True,
            no_nl=True
        )

        model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_batch_inputs(
            batch=nl_raw,
            vocab=nl_vocab,
            processor=Vocab.sos_processor,
            max_len=args.max_nl_len
        )
        model_inputs['labels'], _ = get_batch_inputs(batch=nl_raw,
                                                     vocab=nl_vocab,
                                                     processor=Vocab.eos_processor,
                                                     max_len=args.max_nl_len)
    # summarization
    elif task == enums.TASK_SUMMARIZATION:

        code_raw, nl_raw = map(list, zip(*batch))

        model_inputs['input_ids'], model_inputs['attention_mask'] = get_concat_batch_inputs(
            code_raw=code_raw,
            nl_raw=nl_raw,
            ast_raw=None,
            code_vocab=code_vocab,
            max_code_len=args.max_code_len,
            ast_vocab=ast_vocab,
            max_ast_len=args.max_ast_len,
            nl_vocab=None,
            max_nl_len=args.max_nl_len,
            no_ast=True,
            no_nl=True
        )

        model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_batch_inputs(
            batch=nl_raw,
            vocab=nl_vocab,
            processor=Vocab.sos_processor,
            max_len=args.max_nl_len,
        )

        model_inputs['labels'], _ = get_batch_inputs(
            batch=nl_raw,
            vocab=nl_vocab,
            processor=Vocab.eos_processor,
            max_len=args.max_nl_len,
        )
    # clone detection
    elif task == enums.TASK_CLONE_DETECTION:
        pass
        # code_1_raw, ast_1_raw, name_1_raw, code_2_raw, ast_2_raw, name_2_raw, labels = map(list, zip(*batch))
        #
        # model_inputs['input_ids'], model_inputs['attention_mask'] = get_concat_batch_inputs(
        #     code_raw=code_1_raw,
        #     code_vocab=code_vocab,
        #     max_code_len=args.max_code_len,
        #     ast_raw=ast_1_raw,
        #     ast_vocab=ast_vocab,
        #     max_ast_len=args.max_ast_len,
        #     nl_raw=name_1_raw,
        #     nl_vocab=nl_vocab,
        #     max_nl_len=args.max_nl_len,
        #     no_ast=args.no_ast,
        #     no_nl=args.no_nl
        # )
        # model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_concat_batch_inputs(
        #     code_raw=code_2_raw,
        #     code_vocab=code_vocab,
        #     max_code_len=args.max_code_len,
        #     ast_raw=ast_2_raw,
        #     ast_vocab=ast_vocab,
        #     max_ast_len=args.max_ast_len,
        #     nl_raw=name_2_raw,
        #     nl_vocab=nl_vocab,
        #     max_nl_len=args.max_nl_len,
        #     no_ast=args.no_ast,
        #     no_nl=args.no_nl
        # )
        # model_inputs['labels'] = torch.tensor(labels, dtype=torch.long)


    return model_inputs


def get_batch_inputs(batch: List[str], vocab: Vocab, processor=None, max_len=None):
    """
    Encode the given batch to input to the model.

    Args:
        batch (list[str]): Batch of sequence,
            each sequence is represented by a string or list of tokens
        vocab (Vocab): Vocab of the batch
        processor (tokenizers.processors.PostProcessor): Optional, post-processor method
        max_len (int): Optional, the maximum length of each sequence

    Returns:
        (torch.LongTensor, torch.LongTensor): Tensor of batch and mask, [B, T]

    """
    # set post processor
    vocab.tokenizer.post_processor = processor
    # set truncation
    if max_len:
        vocab.tokenizer.enable_truncation(max_length=max_len)
    else:
        vocab.tokenizer.no_truncation()
    # encode batch
    inputs, padding_mask = vocab.encode_batch(batch, pad=True, max_length=max_len)
    # to tensor
    inputs = torch.tensor(inputs, dtype=torch.long)
    padding_mask = torch.tensor(padding_mask, dtype=torch.long)

    return inputs, padding_mask


def get_concat_batch_inputs(code_raw, code_vocab, max_code_len,
                            ast_raw, ast_vocab, max_ast_len,
                            nl_raw, nl_vocab, max_nl_len,
                            no_ast=False, no_nl=False):
    """
    Return the concat tensor and mask for input.

    Args:
        code_raw:
        code_vocab:
        max_code_len:
        ast_raw:
        ast_vocab:
        max_ast_len:
        nl_raw:
        nl_vocab:
        max_nl_len:
        no_ast:
        no_nl:

    Returns:
        (torch.Tensor, torch.Tensor):
            - Concat inputs
            - concat attention mask

    """
    code_inputs, code_padding_mask = get_batch_inputs(batch=code_raw,
                                                      vocab=code_vocab,
                                                      processor=Vocab.sep_processor,
                                                      max_len=max_code_len)

    if not no_ast:
        ast_inputs, ast_padding_mask = get_batch_inputs(batch=ast_raw,
                                                        vocab=ast_vocab,
                                                        processor=Vocab.sep_processor,
                                                        max_len=max_ast_len)
    else:
        ast_inputs, ast_padding_mask = None, None

    if not no_nl:
        nl_inputs, nl_padding_mask = get_batch_inputs(batch=nl_raw,
                                                      vocab=nl_vocab,
                                                      processor=Vocab.eos_processor,
                                                      max_len=max_nl_len)
    else:
        nl_inputs, nl_padding_mask = None, None

    inputs = torch.cat([inputs for inputs in [code_inputs, ast_inputs, nl_inputs] if inputs is not None], dim=-1)
    padding_mask = torch.cat([mask for mask in [code_padding_mask, ast_padding_mask, nl_padding_mask]
                              if mask is not None], dim=-1)

    # code_inputs, code_padding_mask = get_batch_inputs(batch=code_raw,
    #                                                   vocab=code_vocab,
    #                                                   processor=Vocab.sep_processor,
    #                                                   max_len=max_code_len)
    # ast_inputs, ast_padding_mask = get_batch_inputs(batch=ast_raw,
    #                                                 vocab=ast_vocab,
    #                                                 processor=Vocab.sep_processor,
    #                                                 max_len=max_ast_len)
    # nl_inputs, nl_padding_mask = get_batch_inputs(batch=nl_raw,
    #                                               vocab=nl_vocab,
    #                                               processor=Vocab.eos_processor,
    #                                               max_len=max_nl_len)
    #
    # inputs = torch.cat([code_inputs, ast_inputs, nl_inputs], dim=-1)
    # padding_mask = torch.cat([code_padding_mask, ast_padding_mask, nl_padding_mask], dim=-1)

    return inputs, padding_mask


def pad_batch(batch, pad_value=0):
    """
    Pad a list of sequence to a padded 2d tensor.

    Args:
        batch (list[list[int]]): List of sequence
        pad_value (int): Optional, fill value, default to 0.

    Returns:
        torch.Tensor: Padded tensor. [B, T].

    """
    batch = list(zip(*itertools.zip_longest(*batch, fillvalue=pad_value)))
    return torch.tensor([list(b) for b in batch]).long()

def get_batch_entity_ids(args, input_tokens, entity_dict):
    batch_tokens = []
    word_masks = []
    for item in input_tokens:
        item_tokens = item.split()
        item_token, word_mask = get_entity_ids(args, item_tokens, entity_dict)
        batch_tokens.append(item_token)
        word_masks.append(word_mask)

    batch_tokens = torch.tensor(batch_tokens)
    word_masks = torch.tensor(word_masks)
    return batch_tokens, word_masks

def get_entity_ids(args, input_token, entity_dict):
    input_entity_ids = []
    word_mask = []
    no_match_entity_id = 0
    for t in input_token:
        pt = remove_special_characters(t)
        if pt in entity_dict.keys():
            tid = entity_dict[pt]
            input_entity_ids.append(tid)
        else:
            tid = match_expose_token(pt, entity_dict)
            if tid is not None:
                input_entity_ids.append(tid)
            else:
                input_entity_ids.append(no_match_entity_id)
    if len(input_entity_ids) < args.max_code_len:
        n_pad = args.max_code_len - len(input_entity_ids)
        input_entity_ids.extend([0] * n_pad)

        word_mask = [1] * (args.max_code_len - n_pad)
        word_mask.extend([0] * n_pad)
    else:
        input_entity_ids = input_entity_ids[:args.max_code_len]
        word_mask = [1] * len(input_entity_ids)

    return input_entity_ids, word_mask

def match_expose_token(token, entity_dict):
    n_l = split_camel(token)
    token_ids = []
    for word in n_l:
        if word in entity_dict.keys():
            tid = entity_dict[word]
            token_ids.append(tid)
    if len(token_ids) > 0:
        return token_ids[len(token_ids)-1]
    else:
        return None

def remove_special_characters(input_string):
    # Use a regular expression to remove all non-alphanumeric characters
    return re.sub(r'[^a-zA-Z0-9]', '', input_string)

def split_camel(phrase):
    # Split the phrase based on camel case
    split_phrase = re.findall(r'[A-Z](?:[a-z]+|$)', phrase)
    return split_phrase