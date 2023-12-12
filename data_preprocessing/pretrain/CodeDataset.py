import math
import os
import random
import re

from torch.utils.data.dataset import Dataset

import enums
from data_preprocessing.pretrain.data_utils import load_dataset_from_dir
from data_preprocessing.pretrain.vocab import load_vocab, init_vocab, Vocab
from model.configuration_bart import BartConfig
from util import batch_list_to_batch_tensors


class CodeDataset(Dataset):

    def __init__(self, args, logger, entity_dict, dataset_type, task='mass', language=None):
        """
        :param path:dataset dir path
        :param task: [mass(Masked Span Prediction), nsp(Natural Language Prediction)]
        :param language:
        """
        super(CodeDataset, self).__init__()
        self.args = args
        self.paths = {}
        self.task = task
        self.language = language
        self.logger = logger
        self.paths, self.languages, self.all_sources, self.all_codes, self.all_docs = load_dataset_from_dir(dataset_dir=args.dataset_dir, dataset_type=dataset_type)
        self.code_tokenizer, self.nl_tokenizer = self.get_tokenizers()

        entity2id = os.path.join(args.kg_path, 'entity2id.txt')
        rel2id = os.path.join(args.kg_path, 'relation2id.txt')
        train2id = os.path.join(args.kg_path, 'train2id.txt')

        # self.all_sources = random.sample(self.all_sources, int(len(self.all_sources) / 100))
        # self.all_codes = random.sample(self.all_codes, int(len(self.all_codes) / 100))
        # self.all_docs = random.sample(self.all_docs, int(len(self.all_docs) / 100))


        self.entity_dict = entity_dict
        self.no_match_entity_id = 0

    def __len__(self):
        return len(self.all_codes)

    def match_expose_token(self, token):
        n_l = self.split_camel(token)
        token_ids = []
        for word in n_l:
            if word in self.entity_dict.keys():
                tid = self.entity_dict[word]
                token_ids.append(tid)
        if len(token_ids) > 0:
            return token_ids[len(token_ids)-1]
        else:
            return None

    def split_camel(self, phrase):
        # Split the phrase based on camel case
        split_phrase = re.findall(r'[A-Z](?:[a-z]+|$)', phrase)
        return split_phrase

    def remove_special_characters(self, input_string):
        # Use a regular expression to remove all non-alphanumeric characters
        return re.sub(r'[^a-zA-Z0-9]', '', input_string)

    def get_entity_ids(self, input_tokens):
        input_entity_ids = []
        for t in input_tokens:
            pt = self.remove_special_characters(t)
            if pt in self.entity_dict.keys():
                tid = self.entity_dict[pt]
                input_entity_ids.append(tid)
            else:
                tid = self.match_expose_token(pt)
                if tid is not None:
                    input_entity_ids.append(tid)
                else:
                    input_entity_ids.append(self.no_match_entity_id)

        # for t in input_tokens:
        #     input_entity_ids.append(0)
        return input_entity_ids

    def __getitem__(self, index):
        if self.task == enums.TASK_MASS:
            code_tokens = self.all_codes[index].split()
            mask_len = int(self.args.mass_mask_ratio * len(code_tokens))
            mask_start = random.randint(0, len(code_tokens) - mask_len)
            mask_tokens = code_tokens[mask_start: mask_start + mask_len]
            input_tokens = code_tokens[:mask_start] + [Vocab.MSK_TOKEN] + code_tokens[mask_start + mask_len:]

            # input_ids, input_entity_ids, subword_mask, word_mask, word_subword, decoder_input_ids, decoder_attention_mask, labels
            input_ids, encoder_attention_mask = self.code_tokenizer.encode_sequence(input_tokens, is_pre_tokenized=True, max_len=self.args.input_max_len)

            input_entity_ids = self.get_entity_ids(input_tokens)
            ie_max_len = self.args.input_max_len -2
            if len(input_entity_ids) < ie_max_len:
                n_pad = self.args.input_max_len - len(input_entity_ids) - 1
                word_mask = [1] * (self.args.input_max_len - n_pad)
                word_mask.extend([0] * n_pad)
                input_entity_ids = [1] + input_entity_ids + [2]
                input_entity_ids.extend([0] * (n_pad - 1))
            else:
                input_entity_ids = input_entity_ids[:ie_max_len]
                input_entity_ids = [1] + input_entity_ids + [2]
                word_mask = [1] * len(input_entity_ids)

            decoder_input_ids, decoder_attention_mask = self.code_tokenizer.encode_sequence(code_tokens, is_pre_tokenized=True, max_len=self.args.output_max_len)
            labels, labels_mask = self.code_tokenizer.encode_sequence(code_tokens, is_pre_tokenized=True, max_len=self.args.output_max_len)

            return input_ids, encoder_attention_mask, input_entity_ids, word_mask, decoder_input_ids, decoder_attention_mask, labels

            # return ' '.join(input_tokens), self.asts[index], self.names[index], ' '.join(mask_tokens)
        elif self.task == enums.TASK_NSP:
            code_tokens = self.all_codes[index].split()
            nl_tokens = self.all_docs[index].split()

            input_ids, encoder_attention_mask = self.code_tokenizer.encode_sequence(code_tokens, is_pre_tokenized=True, max_len=self.args.input_max_len)
            input_entity_ids = self.get_entity_ids(code_tokens)
            ie_max_len = self.args.input_max_len - 2
            if len(input_entity_ids) < ie_max_len:
                n_pad = self.args.input_max_len - len(input_entity_ids) - 1
                word_mask = [1] * (self.args.input_max_len - n_pad)
                word_mask.extend([0] * n_pad)
                input_entity_ids = [1] + input_entity_ids + [2]
                input_entity_ids.extend([0] * (n_pad - 1))
            else:
                input_entity_ids = input_entity_ids[:ie_max_len]
                input_entity_ids = [1] + input_entity_ids + [2]
                word_mask = [1] * len(input_entity_ids)

            decoder_input_ids, decoder_attention_mask = self.nl_tokenizer.encode_sequence(nl_tokens, is_pre_tokenized=True, max_len=self.args.output_max_len)
            labels, labels_mask = self.nl_tokenizer.encode_sequence(nl_tokens, is_pre_tokenized=True, max_len=self.args.output_max_len)

            return input_ids, encoder_attention_mask, input_entity_ids, word_mask, decoder_input_ids, decoder_attention_mask, labels

    def __iter__(self):  # iterator to load data
        for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
            batch = []
            for __ in range(self.batch_size):
                idx = random.randint(0, len(self.ex_list) - 1)
                batch.append(self.__getitem__(idx))
            # To Tensor
            yield batch_list_to_batch_tensors(batch)

    def get_tokenizers(self):
        # --------------------------------------------------
        # vocabs
        # --------------------------------------------------
        trained_vocab = self.args.trained_vocab

        self.logger.info('-' * 100)
        if os.path.exists(trained_vocab):
            self.logger.info('Loading vocabularies from files')
            code_vocab = load_vocab(vocab_root=trained_vocab, name=self.args.code_vocab_name)
            nl_vocab = load_vocab(vocab_root=trained_vocab, name=self.args.nl_vocab_name)
        else:
            self.logger.info('Building vocabularies')
            # code vocab
            code_vocab = init_vocab(vocab_save_dir=self.args.vocab_save_dir,
                                    name=self.args.code_vocab_name,
                                    method=self.args.code_tokenize_method,
                                    vocab_size=self.args.code_vocab_size,
                                    datasets=[self.all_codes],
                                    ignore_case=False,
                                    save_root=self.args.vocab_root
                                    )
            # nl vocab
            nl_vocab = init_vocab(vocab_save_dir=self.args.vocab_save_dir,
                                  name=self.args.nl_vocab_name,
                                  method=self.args.nl_tokenize_method,
                                  vocab_size=self.args.nl_vocab_size,
                                  datasets=[self.all_docs],
                                  ignore_case=False,
                                  save_root=self.args.vocab_root,
                                  index_offset=len(code_vocab)
                                  )

        self.logger.info(f'The size of code vocabulary: {len(code_vocab)}')
        self.logger.info(f'The size of nl vocabulary: {len(nl_vocab)}')
        self.logger.info('Vocabularies built successfully')
        return code_vocab, nl_vocab
        # --------------------------------------------------

    def get_vocab_size(self):
        return len(self.code_tokenizer)+len(self.nl_tokenizer)