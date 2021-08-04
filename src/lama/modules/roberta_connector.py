# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from fairseq.models.roberta import RobertaModel
from fairseq import utils
import torch
from pytorch_pretrained_bert import BertTokenizer
from lama.modules.base_connector import *
from lama.modules.bert_connector import CustomBaseTokenizer


class RobertaVocab(object):
    def __init__(self, roberta):
        self.roberta = roberta

    def __getitem__(self, arg):
        value = ""
        try:
            predicted_token_bpe = self.roberta.task.source_dictionary.string([arg])
            if (
                predicted_token_bpe.strip() == ROBERTA_MASK
                or predicted_token_bpe.strip() == ROBERTA_START_SENTENCE
            ):
                value = predicted_token_bpe.strip()
            else:
                value = self.roberta.bpe.decode(str(predicted_token_bpe)).strip()
        except Exception as e:
            print(arg)
            print(predicted_token_bpe)
            print(value)
            print("Exception {} for input {}".format(e, arg))
        return value


class Roberta(Base_Connector):
    def __init__(self, args):
        super().__init__()
        roberta_model_dir = args.roberta_model_dir
        roberta_model_name = args.roberta_model_name
        roberta_vocab_name = args.roberta_vocab_name
        self.dict_file = "{}/{}".format(roberta_model_dir, roberta_vocab_name)
        self.model = RobertaModel.from_pretrained(
            roberta_model_dir, checkpoint_file=roberta_model_name
        )

        self.bpe = self.model.bpe
        self.task = self.model.task
        self._build_vocab()
        self._init_inverse_vocab()
        self.max_sentence_length = args.max_sentence_length

        # CD: Add custom tokenizer to avoid splitting the ['MASK'] token
        # self.tokenizer = BertTokenizer.from_pretrained(dict_file)
        # custom_basic_tokenizer = CustomBaseTokenizer(do_lower_case = do_lower_case)
        # self.tokenizer.basic_tokenizer = custom_basic_tokenizer

    def _cuda(self):
        self.model.cuda()

    def _build_vocab(self):
        self.vocab = []
        for key in range(ROBERTA_VOCAB_SIZE):
            predicted_token_bpe = self.task.source_dictionary.string([key])
            try:
                value = self.bpe.decode(predicted_token_bpe)

                if value[0] == " ":  # if the token starts with a whitespace
                    value = value.strip()
                else:
                    # this is subword information
                    value = "_{}_".format(value)

                if value in self.vocab:
                    # print("WARNING: token '{}' is already in the vocab".format(value))
                    value = "{}_{}".format(value, key)

                self.vocab.append(value)

            except Exception as e:
                self.vocab.append(predicted_token_bpe.strip())

    def get_id(self, input_string):
        # Roberta predicts ' London' and not 'London'
        string = " " + str(input_string).strip()
        text_spans_bpe = self.bpe.encode(string.rstrip())
        tokens = self.task.source_dictionary.encode_line(
            text_spans_bpe, append_eos=False
        )
        return tokens.long()

    def get_batch_generation(self, sentences_list, logger=None, try_cuda=True):
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tensor_list = []
        masked_indices_list = []
        max_len = 0
        output_tokens_list = []
        for masked_inputs_list in sentences_list:

            tokens_list = []

            for idx, masked_input in enumerate(masked_inputs_list):

                # 2. sobstitute [MASK] with <mask>
                masked_input = masked_input.replace(MASK, ROBERTA_MASK)

                text_spans = masked_input.split(ROBERTA_MASK)
                text_spans_bpe = (
                    (" {0} ".format(ROBERTA_MASK))
                    .join(
                        [
                            self.bpe.encode(text_span.rstrip())
                            for text_span in text_spans
                        ]
                    )
                    .strip()
                )

                prefix = ""
                if idx == 0:
                    prefix = ROBERTA_START_SENTENCE

                tokens_list.append(
                    self.task.source_dictionary.encode_line(
                        prefix + " " + text_spans_bpe, append_eos=True
                    )
                )

            tokens = torch.cat(tokens_list)[: self.max_sentence_length]
            output_tokens_list.append(tokens.long().cpu().numpy())

            if len(tokens) > max_len:
                max_len = len(tokens)
            tensor_list.append(tokens)
            masked_index = (tokens == self.task.mask_idx).nonzero().numpy()
            for x in masked_index:
                masked_indices_list.append([x[0]])

        pad_id = self.task.source_dictionary.pad()
        tokens_list = []
        for tokens in tensor_list:
            pad_lenght = max_len - len(tokens)
            if pad_lenght > 0:
                pad_tensor = torch.full([pad_lenght], pad_id, dtype=torch.int)
                tokens = torch.cat((tokens, pad_tensor))
            tokens_list.append(tokens)

        batch_tokens = torch.stack(tokens_list)

        with torch.no_grad():
            # with utils.eval(self.model.model):
            self.model.eval()
            self.model.model.eval()
            log_probs, extra = self.model.model(
                batch_tokens.long().to(device=self._model_device),
                features_only=False,
                return_all_hiddens=False,
            )

        return log_probs.cpu(), output_tokens_list, masked_indices_list

    def __get_input_tensors(self, sentences):

        if len(sentences) > 2:
            print(sentences)
            raise ValueError("BERT accepts maximum two sentences in input for each data point")

        first_tokenized_sentence = self.tokenizer.tokenize(sentences[0])
        first_segment_id = np.zeros(len(first_tokenized_sentence), dtype=int).tolist()

        # add [SEP] token at the end
        first_tokenized_sentence.append(BERT_SEP)
        first_segment_id.append(0)

        if len(sentences)>1 :
            second_tokenized_sentece = self.tokenizer.tokenize(sentences[1])
            second_segment_id = np.full(len(second_tokenized_sentece),1, dtype=int).tolist()

            # add [SEP] token at the end
            second_tokenized_sentece.append(BERT_SEP)
            second_segment_id.append(1)

            tokenized_text = first_tokenized_sentence + second_tokenized_sentece
            segments_ids = first_segment_id + second_segment_id
        else:
            tokenized_text = first_tokenized_sentence
            segments_ids = first_segment_id

        # add [CLS] token at the beginning
        tokenized_text.insert(0,BERT_CLS)
        segments_ids.insert(0,0)

        # look for masked indices
        masked_indices = []
        for i in range(len(tokenized_text)):
            token = tokenized_text[i]
            if token == MASK:
                masked_indices.append(i)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        return tokens_tensor, segments_tensors, masked_indices, tokenized_text


    def __get_input_tensors_batch(self, sentences_list):
        tokens_tensors_list = []
        segments_tensors_list = []
        masked_indices_list = []
        tokenized_text_list = []
        max_tokens = 0
        for sentences in sentences_list:
            tokens_tensor, segments_tensor, masked_indices, tokenized_text = self.__get_input_tensors(sentences)
            tokens_tensors_list.append(tokens_tensor)
            segments_tensors_list.append(segments_tensor)
            masked_indices_list.append(masked_indices)
            tokenized_text_list.append(tokenized_text)
            # assert(tokens_tensor.shape[1] == segments_tensor.shape[1])
            if (tokens_tensor.shape[1] > max_tokens):
                max_tokens = tokens_tensor.shape[1]
        # print("MAX_TOKENS: {}".format(max_tokens))
        # apply padding and concatenate tensors
        # use [PAD] for tokens and 0 for segments
        final_tokens_tensor = None
        final_segments_tensor = None
        final_attention_mask = None
        for tokens_tensor, segments_tensor in zip(tokens_tensors_list, segments_tensors_list):
            dim_tensor = tokens_tensor.shape[1]
            pad_lenght = max_tokens - dim_tensor
            attention_tensor = torch.full([1,dim_tensor], 1, dtype= torch.long)
            if pad_lenght>0:
                pad_1 = torch.full([1,pad_lenght], self.pad_id, dtype= torch.long)
                pad_2 = torch.full([1,pad_lenght], 0, dtype= torch.long)
                attention_pad = torch.full([1,pad_lenght], 0, dtype= torch.long)
                tokens_tensor = torch.cat((tokens_tensor,pad_1), dim=1)
                segments_tensor = torch.cat((segments_tensor,pad_2), dim=1)
                attention_tensor = torch.cat((attention_tensor,attention_pad), dim=1)
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_segments_tensor = segments_tensor
                final_attention_mask = attention_tensor
            else:
                final_tokens_tensor = torch.cat((final_tokens_tensor,tokens_tensor), dim=0)
                final_segments_tensor = torch.cat((final_segments_tensor,segments_tensor), dim=0)
                final_attention_mask = torch.cat((final_attention_mask,attention_tensor), dim=0)
        # print(final_tokens_tensor)
        # print(final_segments_tensor)
        # print(final_attention_mask)
        # print(final_tokens_tensor.shape)
        # print(final_segments_tensor.shape)
        # print(final_attention_mask.shape)
        return final_tokens_tensor, final_segments_tensor, final_attention_mask, masked_indices_list, tokenized_text_list

    def get_contextual_embeddings_with_mask_indices(self, sentences_list, try_cuda=True):
        # TBA
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tensor_list = []
        masked_indices_list = []
        max_len = 0
        output_tokens_list = []
        for masked_inputs_list in sentences_list:

            tokens_list = []

            for idx, masked_input in enumerate(masked_inputs_list):

                # 2. sobstitute [MASK] with <mask>
                masked_input = masked_input.replace(MASK, ROBERTA_MASK)

                text_spans = masked_input.split(ROBERTA_MASK)
                text_spans_bpe = (
                    (" {0} ".format(ROBERTA_MASK))
                    .join(
                        [
                            self.bpe.encode(text_span.rstrip())
                            for text_span in text_spans
                        ]
                    )
                    .strip()
                )

                prefix = ""
                if idx == 0:
                    prefix = ROBERTA_START_SENTENCE

                tokens_list.append(
                    self.task.source_dictionary.encode_line(
                        prefix + " " + text_spans_bpe, append_eos=True
                    )
                )

            tokens = torch.cat(tokens_list)[: self.max_sentence_length]
            output_tokens_list.append(tokens.long().cpu().numpy())

            if len(tokens) > max_len:
                max_len = len(tokens)
            tensor_list.append(tokens)
            masked_index = (tokens == self.task.mask_idx).nonzero().numpy()
            for x in masked_index:
                masked_indices_list.append([x[0]])

        pad_id = self.task.source_dictionary.pad()
        tokens_list = []
        for tokens in tensor_list:
            pad_lenght = max_len - len(tokens)
            if pad_lenght > 0:
                pad_tensor = torch.full([pad_lenght], pad_id, dtype=torch.int)
                tokens = torch.cat((tokens, pad_tensor))
            tokens_list.append(tokens)

        batch_tokens = torch.stack(tokens_list)

        with torch.no_grad():
            # with utils.eval(self.model.model):
            self.model.eval()
            self.model.model.eval()
            log_probs, extra = self.model.model(
                batch_tokens.long().to(device=self._model_device),
                features_only=True,
                return_all_hiddens=False,
            )

        # return log_probs.cpu(), output_tokens_list, masked_indices_list
        return [log_probs.cpu()], None, None, masked_indices_list
        # self.get_batch_generation(sentences_list)

        # # assume in input 1 or 2 sentences - in general, it considers only the first 2 sentences
        # if not sentences_list:
        #     return None
        # if try_cuda:
        #     self.try_cuda()

        # tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, tokenized_text_list = self.__get_input_tensors_batch(sentences_list)

        # with torch.no_grad():
        #     all_encoder_layers, _ = self.bert_model(
        #         tokens_tensor.to(self._model_device),
        #         segments_tensor.to(self._model_device))

        # all_encoder_layers = [layer.cpu() for layer in all_encoder_layers]

        # sentence_lengths = [len(x) for x in tokenized_text_list]

        # # all_encoder_layers: a list of the full sequences of encoded-hidden-states at the end
        # # of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
        # # encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size]
        # return all_encoder_layers, sentence_lengths, tokenized_text_list

        # # return None
