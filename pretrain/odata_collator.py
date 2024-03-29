
import torch

from typing import List
import itertools


import enums


def collate_fn(batch, args, task):
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

        input_ids, encoder_attention_mask, input_entity_ids, word_mask, decoder_input_ids, decoder_attention_mask, labels = map(list, zip(*batch))

        # model_inputs['input_ids'], model_inputs['attention_mask'] = get_concat_batch_inputs(
        #     code_raw=code_raw,
        #     code_vocab=code_vocab,
        #     max_code_len=args.max_code_len,
        #     ast_raw=ast_raw,
        #     ast_vocab=ast_vocab,
        #     max_ast_len=args.max_ast_len,
        #     nl_raw=name_raw,
        #     nl_vocab=nl_vocab,
        #     max_nl_len=args.max_nl_len,
        #     no_ast=args.no_ast,
        #     no_nl=args.no_nl
        # )
        # model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_batch_inputs(
        #     batch=target_raw,
        #     vocab=code_vocab,
        #     processor=Vocab.sos_processor,
        #     max_len=int(args.mass_mask_ratio * args.max_code_len)
        # )
        # model_inputs['labels'], _ = get_batch_inputs(batch=target_raw,
        #                                              vocab=code_vocab,
        #                                              processor=Vocab.eos_processor,
        #                                              max_len=int(args.mass_mask_ratio * args.max_code_len))

        model_inputs['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
        model_inputs['attention_mask'] = torch.tensor(encoder_attention_mask, dtype=torch.long)
        model_inputs['decoder_input_ids'] = torch.tensor(decoder_input_ids, dtype=torch.long)
        model_inputs['decoder_attention_mask'] = torch.tensor(decoder_attention_mask, dtype=torch.long)
        model_inputs['labels'] = torch.tensor(labels, dtype=torch.long)


    # summarization
    elif task == enums.TASK_NSP:
        input_ids, encoder_attention_mask, input_entity_ids, word_mask, decoder_input_ids, decoder_attention_mask, labels = map(list, zip(*batch))
        model_inputs['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
        model_inputs['attention_mask'] = torch.tensor(encoder_attention_mask, dtype=torch.long)
        model_inputs['decoder_input_ids'] = torch.tensor(decoder_input_ids, dtype=torch.long)
        model_inputs['decoder_attention_mask'] = torch.tensor(decoder_attention_mask, dtype=torch.long)
        model_inputs['labels'] = torch.tensor(labels, dtype=torch.long)
        # code_raw, ast_raw, name_raw, nl_raw = map(list, zip(*batch))
        #
        # model_inputs['input_ids'], model_inputs['attention_mask'] = get_concat_batch_inputs(
        #     code_raw=code_raw,
        #     code_vocab=code_vocab,
        #     max_code_len=args.max_code_len,
        #     ast_raw=ast_raw,
        #     ast_vocab=ast_vocab,
        #     max_ast_len=args.max_ast_len,
        #     nl_raw=name_raw,
        #     nl_vocab=nl_vocab,
        #     max_nl_len=args.max_nl_len,
        #     no_ast=args.no_ast,
        #     no_nl=args.no_nl
        # )
        #
        # model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_batch_inputs(
        #     batch=nl_raw,
        #     vocab=nl_vocab,
        #     processor=Vocab.sos_processor,
        #     max_len=args.max_nl_len,
        # )
        #
        # model_inputs['labels'], _ = get_batch_inputs(
        #     batch=nl_raw,
        #     vocab=nl_vocab,
        #     processor=Vocab.eos_processor,
        #     max_len=args.max_nl_len,
        # )
    elif task == enums.TASK_SUMMARIZATION:
        input_ids, encoder_attention_mask, input_entity_ids, word_mask, decoder_input_ids, decoder_attention_mask, labels = map(
            list, zip(*batch))
        model_inputs['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
        model_inputs['attention_mask'] = torch.tensor(encoder_attention_mask, dtype=torch.long)
        model_inputs['decoder_input_ids'] = torch.tensor(decoder_input_ids, dtype=torch.long)
        model_inputs['decoder_attention_mask'] = torch.tensor(decoder_attention_mask, dtype=torch.long)
        model_inputs['labels'] = torch.tensor(labels, dtype=torch.long)
    # clone detection
    # elif task == enums.TASK_CLONE_DETECTION:
    #     code_1_raw, ast_1_raw, name_1_raw, code_2_raw, ast_2_raw, name_2_raw, labels = map(list, zip(*batch))
    #
    #     model_inputs['input_ids'], model_inputs['attention_mask'] = get_concat_batch_inputs(
    #         code_raw=code_1_raw,
    #         code_vocab=code_vocab,
    #         max_code_len=args.max_code_len,
    #         ast_raw=ast_1_raw,
    #         ast_vocab=ast_vocab,
    #         max_ast_len=args.max_ast_len,
    #         nl_raw=name_1_raw,
    #         nl_vocab=nl_vocab,
    #         max_nl_len=args.max_nl_len,
    #         no_ast=args.no_ast,
    #         no_nl=args.no_nl
    #     )
    #     model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_concat_batch_inputs(
    #         code_raw=code_2_raw,
    #         code_vocab=code_vocab,
    #         max_code_len=args.max_code_len,
    #         ast_raw=ast_2_raw,
    #         ast_vocab=ast_vocab,
    #         max_ast_len=args.max_ast_len,
    #         nl_raw=name_2_raw,
    #         nl_vocab=nl_vocab,
    #         max_nl_len=args.max_nl_len,
    #         no_ast=args.no_ast,
    #         no_nl=args.no_nl
    #     )
    #     model_inputs['labels'] = torch.tensor(labels, dtype=torch.long)

    return model_inputs


# def get_batch_inputs(batch: List[str], vocab: Vocab, processor=None, max_len=None):
#     """
#     Encode the given batch to input to the model.
#
#     Args:
#         batch (list[str]): Batch of sequence,
#             each sequence is represented by a string or list of tokens
#         vocab (Vocab): Vocab of the batch
#         processor (tokenizers.processors.PostProcessor): Optional, post-processor method
#         max_len (int): Optional, the maximum length of each sequence
#
#     Returns:
#         (torch.LongTensor, torch.LongTensor): Tensor of batch and mask, [B, T]
#
#     """
#     # set post processor
#     vocab.tokenizer.post_processor = processor
#     # set truncation
#     if max_len:
#         vocab.tokenizer.enable_truncation(max_length=max_len)
#     else:
#         vocab.tokenizer.no_truncation()
#     # encode batch
#     inputs, padding_mask = vocab.encode_batch(batch, pad=True, max_length=max_len)
#     # to tensor
#     inputs = torch.tensor(inputs, dtype=torch.long)
#     padding_mask = torch.tensor(padding_mask, dtype=torch.long)
#
#     return inputs, padding_mask
#
#
# def get_concat_batch_inputs(code_raw, code_vocab, max_code_len,
#                             ast_raw, ast_vocab, max_ast_len,
#                             nl_raw, nl_vocab, max_nl_len,
#                             no_ast=False, no_nl=False):
#     """
#     Return the concat tensor and mask for input.
#
#     Args:
#         code_raw:
#         code_vocab:
#         max_code_len:
#         ast_raw:
#         ast_vocab:
#         max_ast_len:
#         nl_raw:
#         nl_vocab:
#         max_nl_len:
#         no_ast:
#         no_nl:
#
#     Returns:
#         (torch.Tensor, torch.Tensor):
#             - Concat inputs
#             - concat attention mask
#
#     """
#     code_inputs, code_padding_mask = get_batch_inputs(batch=code_raw,
#                                                       vocab=code_vocab,
#                                                       processor=Vocab.sep_processor,
#                                                       max_len=max_code_len)
#
#     if not no_ast:
#         ast_inputs, ast_padding_mask = get_batch_inputs(batch=ast_raw,
#                                                         vocab=ast_vocab,
#                                                         processor=Vocab.sep_processor,
#                                                         max_len=max_ast_len)
#     else:
#         ast_inputs, ast_padding_mask = None, None
#
#     if not no_nl:
#         nl_inputs, nl_padding_mask = get_batch_inputs(batch=nl_raw,
#                                                       vocab=nl_vocab,
#                                                       processor=Vocab.eos_processor,
#                                                       max_len=max_nl_len)
#     else:
#         nl_inputs, nl_padding_mask = None, None
#
#     inputs = torch.cat([inputs for inputs in [code_inputs, ast_inputs, nl_inputs] if inputs is not None], dim=-1)
#     padding_mask = torch.cat([mask for mask in [code_padding_mask, ast_padding_mask, nl_padding_mask]
#                               if mask is not None], dim=-1)
#
#     # code_inputs, code_padding_mask = get_batch_inputs(batch=code_raw,
#     #                                                   vocab=code_vocab,
#     #                                                   processor=Vocab.sep_processor,
#     #                                                   max_len=max_code_len)
#     # ast_inputs, ast_padding_mask = get_batch_inputs(batch=ast_raw,
#     #                                                 vocab=ast_vocab,
#     #                                                 processor=Vocab.sep_processor,
#     #                                                 max_len=max_ast_len)
#     # nl_inputs, nl_padding_mask = get_batch_inputs(batch=nl_raw,
#     #                                               vocab=nl_vocab,
#     #                                               processor=Vocab.eos_processor,
#     #                                               max_len=max_nl_len)
#     #
#     # inputs = torch.cat([code_inputs, ast_inputs, nl_inputs], dim=-1)
#     # padding_mask = torch.cat([code_padding_mask, ast_padding_mask, nl_padding_mask], dim=-1)
#
#     return inputs, padding_mask
#
#
# def pad_batch(batch, pad_value=0):
#     """
#     Pad a list of sequence to a padded 2d tensor.
#
#     Args:
#         batch (list[list[int]]): List of sequence
#         pad_value (int): Optional, fill value, default to 0.
#
#     Returns:
#         torch.Tensor: Padded tensor. [B, T].
#
#     """
#     batch = list(zip(*itertools.zip_longest(*batch, fillvalue=pad_value)))
#     return torch.tensor([list(b) for b in batch]).long()
