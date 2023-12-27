import math
import os
import sys
import random
import re
from torch.utils.data.dataset import Dataset
from data_preprocessing.pretrain.vocab import load_vocab, init_vocab, Vocab
from util import batch_list_to_batch_tensors


class TLDataset(Dataset):

    def __init__(self, args, logger, entity_dict, dataset_type, language=None):
        """
        :param path:dataset dir path
        :param task: [mass(Masked Span Prediction), nsp(Natural Language Prediction)]
        :param language:
        """
        super(TLDataset, self).__init__()
        self.args = args
        self.paths = {}
        self.language = language
        self.logger = logger

        load_path = os.path.join(args.dataset_dir, dataset_type)
        self.all_codes, self.all_docs = self.load_tl_dataset_from_dir(dataset_dir=load_path)
        self.code_tokenizer, self.nl_tokenizer = self.get_tokenizers()

        # self.all_codes = random.sample(self.all_codes, int(len(self.all_codes) / 10))
        # self.all_docs = random.sample(self.all_docs, int(len(self.all_docs) / 10))

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
        code_tokens = self.all_codes[index].split()
        nl_tokens = self.all_docs[index].split()

        input_ids, encoder_attention_mask = self.code_tokenizer.encode_sequence(code_tokens, is_pre_tokenized=True,
                                                                                max_len=self.args.input_max_len)
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

        decoder_input_ids, decoder_attention_mask = self.nl_tokenizer.encode_sequence(nl_tokens, is_pre_tokenized=True,
                                                                                      max_len=self.args.output_max_len)
        labels, labels_mask = self.nl_tokenizer.encode_sequence(nl_tokens, is_pre_tokenized=True,
                                                                max_len=self.args.output_max_len)

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

    def load_tl_dataset_from_dir(self, dataset_dir):
        codes_dict = {}
        all_codes = []
        all_docs = []

        tag = dataset_dir.split('/')
        tag = tag[len(tag) - 1]
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
