import math
import os
import json
import sys
import pickle
import logging
import random
import re
from torch.utils.data.dataset import Dataset

from data_preprocessing.pretrain.data_utils import parse_for_clone
from data_preprocessing.pretrain.vocab import load_vocab, init_vocab, Vocab
from util import batch_list_to_batch_tensors
logger = logging.getLogger(__name__)


class BCLDataset(Dataset):

    def __init__(self, args, logger, entity_dict, dataset_type, language=None):
        """
        :param path:dataset dir path
        :param task: [mass(Masked Span Prediction), nsp(Natural Language Prediction)]
        :param language:
        """
        super(BCLDataset, self).__init__()
        self.args = args
        self.paths = {}
        self.language = language
        self.logger = logger
        self.dataset_type = dataset_type

        load_path = os.path.join(args.dataset_dir, dataset_type)
        # self.all_codes, self.all_docs = self.load_tl_dataset_from_dir(dataset_dir=load_path)

        clone_mapping = self.load_clone_mapping(self.dataset_dir)
        assert clone_mapping
        path = os.path.join(self.dataset_dir, f'{self.dataset_type}.txt')
        self.paths['file'] = path
        self.codes_1, self.asts_1, self.names_1, \
        self.codes_2, self.asts_2, self.names_2, self.labels = parse_for_clone(path=path,
                                                                               mapping=clone_mapping)
        assert len(self.codes_1) == len(self.asts_1) == len(self.names_1) \
               == len(self.codes_2) == len(self.asts_2) == len(self.names_2) == len(self.labels)
        self.size = len(self.codes_1)
        logger.info('The size of load dataset: {}'.format(self.size))



        self.code_tokenizer, self.nl_tokenizer = self.get_tokenizers()

        # self.all_codes = random.sample(self.all_codes, int(len(self.all_codes) / 100))
        # self.all_docs = random.sample(self.all_docs, int(len(self.all_docs) / 100))

        # self.kg_matcher = KGMatcher(
        #     entity2id = entity2id,
        #     rel2id = rel2id,
        #     train2id = train2id,
        # )
        # self.kg_matcher = kg_matcher
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

    def load_clone_mapping(self, path):
        # load index
        logger.info("Creating features from index file at %s ", path)
        url_to_code = {}
        with open(path + '/data.jsonl') as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                url_to_code[js['idx']] = js['func']
        return url_to_code

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
        # code_tokens = self.all_codes[index].split()
        # nl_tokens = self.all_docs[index].split()
        #
        # input_ids, encoder_attention_mask = self.code_tokenizer.encode_sequence(code_tokens, is_pre_tokenized=True,
        #                                                                         max_len=self.args.input_max_len)
        # input_entity_ids = self.get_entity_ids(code_tokens)
        # ie_max_len = self.args.input_max_len - 2
        # if len(input_entity_ids) < ie_max_len:
        #     n_pad = self.args.input_max_len - len(input_entity_ids) - 1
        #     word_mask = [1] * (self.args.input_max_len - n_pad)
        #     word_mask.extend([0] * n_pad)
        #     input_entity_ids = [1] + input_entity_ids + [2]
        #     input_entity_ids.extend([0] * (n_pad - 1))
        # else:
        #     input_entity_ids = input_entity_ids[:ie_max_len]
        #     input_entity_ids = [1] + input_entity_ids + [2]
        #     word_mask = [1] * len(input_entity_ids)
        #
        # decoder_input_ids, decoder_attention_mask = self.nl_tokenizer.encode_sequence(nl_tokens, is_pre_tokenized=True,
        #                                                                               max_len=self.args.output_max_len)
        # labels, labels_mask = self.nl_tokenizer.encode_sequence(nl_tokens, is_pre_tokenized=True,
        #                                                         max_len=self.args.output_max_len)
        #
        # return input_ids, encoder_attention_mask, input_entity_ids, word_mask, decoder_input_ids, decoder_attention_mask, labels

        return self.all_codes[index], self.all_docs[index]


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

    def load_tl_dataset_from_dir(self, dataset_dir):
        codes_dict = {}
        all_codes = []
        all_docs = []

        tag = self.dataset_type
        if tag == 'train':
            code_tokn_f = os.path.join(dataset_dir, "train.token.code")
            nl_tokn_f = os.path.join(dataset_dir, "train.token.nl")
        elif tag == 'valid':
            code_tokn_f = os.path.join(dataset_dir, "valid.token.code")
            nl_tokn_f = os.path.join(dataset_dir, "valid.token.nl")
        else:
            code_tokn_f = os.path.join(dataset_dir, "test.token.code")
            nl_tokn_f = os.path.join(dataset_dir, "test.token.nl")

        with open(code_tokn_f, encoding="utf-8") as f:
            datas = f.readlines()
            for data in datas:
                d_l = data.split("	")
                idx = d_l[0]
                code = d_l[1]
                codes_dict[idx] = code

        with open(nl_tokn_f, encoding="utf-8") as f:
            datas = f.readlines()
            for data in datas:
                d_l = data.split("	")
                idx = d_l[0]
                nl = d_l[1]
                code = codes_dict[idx]

                code = code.replace('\n', '')
                nl = nl.replace('\n', '')

                all_codes.append(code)
                all_docs.append(nl)

        return all_codes, all_docs


def print_paths(paths):
    """
    Print paths.

    Args:
        paths (dict): Dict mapping path group to path string or list of path strings.

    """
    logger.info('Dataset loaded from these files:')
    for key, value in paths.items():
        if isinstance(value, list):
            for v in value:
                logger.info(f'  {key}: {v}')
        else:
            logger.info(f'  {key}: {value}')


def init_dataset(args, mode, task=None, language=None, split=None, clone_mapping=None,
                 load_if_saved=True) -> BCLDataset:
    """
    Find dataset, if the dataset is saved, load and return, else initialize and return.

    Args:
        args (argparse.Namespace): Arguments
        mode (str): Training mode, ``pre_train`` or ``fine_tune``
        task (str): Dataset mode, support pre-training tasks: ['cap', 'mass', 'mnp'],
            and downstream fine-tuning task: ['summarization', 'translation'],
            future support ['completion', 'search', 'clone', 'summarization', 'translation']
        language (str): Only for downstream fine-tuning
        split (str): Only for downstream fine-tuning, support ['train', 'valid', 'test', 'codebase(only for search)']
        clone_mapping (dict[int, str]): Mapping from code id to source code string, use only for clone detection
        load_if_saved (bool): Whether to load the saved instance if it exists, default to True

    Returns:
        CodeDataset: Loaded or initialized dataset

    """
    name = '.'.join([sub_name for sub_name in [mode, language, split] if sub_name is not None])
    if load_if_saved:
        path = os.path.join(args.dataset_save_dir, f'{name}.pk')
        if os.path.exists(path) and os.path.isfile(path):
            logger.info(f'Trying to load saved binary pickle file from: {path}')
            with open(path, mode='rb') as f:
                obj = pickle.load(f)
            assert isinstance(obj, BCLDataset)
            obj.args = args
            logger.info(f'Dataset instance loaded from: {path}')
            print_paths(obj.paths)
            return obj
    dataset = BCLDataset(args=args,
                          dataset_name=name,
                          mode=mode,
                          task=task,
                          language=language,
                          split=split,
                          clone_mapping=clone_mapping)
    dataset.save()
    return dataset
