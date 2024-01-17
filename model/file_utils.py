"""
Utilities for working with the local dataset cache.
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
Copyright by the AllenNLP authors.
"""

import fnmatch
import json
import logging
import os
import re
import shutil
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from functools import partial, wraps
from hashlib import sha256
from pathlib import Path
from typing import Dict, Optional, Union
from urllib.parse import urlparse
from zipfile import ZipFile, is_zipfile

import requests
from filelock import FileLock
from tqdm.auto import tqdm

from . import __version__

FAKE_MODEL_DISCLAIMER = """
    <Tip warning={true}>

    This example uses a random model as the real ones are all very big. To get proper results, you should use
    {real_checkpoint} instead of {fake_checkpoint}. If you get out-of-memory when loading that checkpoint, you can try
    adding `device_map="auto"` in the `from_pretrained` call.

    </Tip>
"""


PT_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer(
    ...     "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
    ... )

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> predicted_token_class_ids = logits.argmax(-1)

    >>> # Note that tokens are classified rather then input words which means that
    >>> # there might be more predicted token classes than words.
    >>> # Multiple token classes might account for the same word
    >>> predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
    >>> predicted_tokens_classes
    {expected_output}

    >>> labels = predicted_token_class_ids
    >>> loss = model(**inputs, labels=labels).loss
    >>> round(loss.item(), 2)
    {expected_loss}
    ```
"""

PT_QUESTION_ANSWERING_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

    >>> inputs = tokenizer(question, text, return_tensors="pt")
    >>> with torch.no_grad():
    ...     outputs = model(**inputs)

    >>> answer_start_index = outputs.start_logits.argmax()
    >>> answer_end_index = outputs.end_logits.argmax()

    >>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    >>> tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
    {expected_output}

    >>> # target is "nice puppet"
    >>> target_start_index = torch.tensor([{qa_target_start_index}])
    >>> target_end_index = torch.tensor([{qa_target_end_index}])

    >>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
    >>> loss = outputs.loss
    >>> round(loss.item(), 2)
    {expected_loss}
    ```
"""

PT_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example of single-label classification:

    ```python
    >>> import torch
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> predicted_class_id = logits.argmax().item()
    >>> model.config.id2label[predicted_class_id]
    {expected_output}

    >>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
    >>> num_labels = len(model.config.id2label)
    >>> model = {model_class}.from_pretrained("{checkpoint}", num_labels=num_labels)

    >>> labels = torch.tensor([1])
    >>> loss = model(**inputs, labels=labels).loss
    >>> round(loss.item(), 2)
    {expected_loss}
    ```

    Example of multi-label classification:

    ```python
    >>> import torch
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", problem_type="multi_label_classification")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

    >>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
    >>> num_labels = len(model.config.id2label)
    >>> model = {model_class}.from_pretrained(
    ...     "{checkpoint}", num_labels=num_labels, problem_type="multi_label_classification"
    ... )

    >>> labels = torch.sum(
    ...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
    ... ).to(torch.float)
    >>> loss = model(**inputs, labels=labels).loss
    ```
"""

PT_MASKED_LM_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors="pt")

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> # retrieve index of {mask}
    >>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    >>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    >>> tokenizer.decode(predicted_token_id)
    {expected_output}

    >>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
    >>> # mask labels of non-{mask} tokens
    >>> labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

    >>> outputs = model(**inputs, labels=labels)
    >>> round(outputs.loss.item(), 2)
    {expected_loss}
    ```
"""

PT_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    >>> outputs = model(**inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

PT_MULTIPLE_CHOICE_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    >>> choice0 = "It is eaten with a fork and a knife."
    >>> choice1 = "It is eaten while held in the hand."
    >>> labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

    >>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="pt", padding=True)
    >>> outputs = model(**{{k: v.unsqueeze(0) for k, v in encoding.items()}}, labels=labels)  # batch size is 1

    >>> # the linear classifier still needs to be trained
    >>> loss = outputs.loss
    >>> logits = outputs.logits
    ```
"""

PT_CAUSAL_LM_SAMPLE = r"""
    Example:

    ```python
    >>> import torch
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    >>> outputs = model(**inputs, labels=inputs["input_ids"])
    >>> loss = outputs.loss
    >>> logits = outputs.logits
    ```
"""

PT_SPEECH_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoProcessor, {model_class}
    >>> import torch
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> processor = AutoProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    >>> with torch.no_grad():
    ...     outputs = model(**inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    >>> list(last_hidden_states.shape)
    {expected_output}
    ```
"""

PT_SPEECH_CTC_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoProcessor, {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> processor = AutoProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits
    >>> predicted_ids = torch.argmax(logits, dim=-1)

    >>> # transcribe speech
    >>> transcription = processor.batch_decode(predicted_ids)
    >>> transcription[0]
    {expected_output}

    >>> inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="pt").input_ids

    >>> # compute loss
    >>> loss = model(**inputs).loss
    >>> round(loss.item(), 2)
    {expected_loss}
    ```
"""

PT_SPEECH_SEQ_CLASS_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoFeatureExtractor, {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> feature_extractor = AutoFeatureExtractor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> predicted_class_ids = torch.argmax(logits, dim=-1).item()
    >>> predicted_label = model.config.id2label[predicted_class_ids]
    >>> predicted_label
    {expected_output}

    >>> # compute loss - target_label is e.g. "down"
    >>> target_label = model.config.id2label[0]
    >>> inputs["labels"] = torch.tensor([model.config.label2id[target_label]])
    >>> loss = model(**inputs).loss
    >>> round(loss.item(), 2)
    {expected_loss}
    ```
"""


PT_SPEECH_FRAME_CLASS_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoFeatureExtractor, {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> feature_extractor = AutoFeatureExtractor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = feature_extractor(dataset[0]["audio"]["array"], return_tensors="pt", sampling_rate=sampling_rate)
    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> probabilities = torch.sigmoid(logits[0])
    >>> # labels is a one-hot array of shape (num_frames, num_speakers)
    >>> labels = (probabilities > 0.5).long()
    >>> labels[0].tolist()
    {expected_output}
    ```
"""


PT_SPEECH_XVECTOR_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoFeatureExtractor, {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> feature_extractor = AutoFeatureExtractor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = feature_extractor(
    ...     [d["array"] for d in dataset[:2]["audio"]], sampling_rate=sampling_rate, return_tensors="pt", padding=True
    ... )
    >>> with torch.no_grad():
    ...     embeddings = model(**inputs).embeddings

    >>> embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

    >>> # the resulting embeddings can be used for cosine similarity-based retrieval
    >>> cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    >>> similarity = cosine_sim(embeddings[0], embeddings[1])
    >>> threshold = 0.7  # the optimal threshold is dataset-dependent
    >>> if similarity < threshold:
    ...     print("Speakers are not the same!")
    >>> round(similarity.item(), 2)
    {expected_output}
    ```
"""

PT_VISION_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoImageProcessor, {model_class}
    >>> import torch
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("huggingface/cats-image")
    >>> image = dataset["test"]["image"][0]

    >>> image_processor = AutoImageProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = image_processor(image, return_tensors="pt")

    >>> with torch.no_grad():
    ...     outputs = model(**inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    >>> list(last_hidden_states.shape)
    {expected_output}
    ```
"""

PT_VISION_SEQ_CLASS_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoImageProcessor, {model_class}
    >>> import torch
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("huggingface/cats-image")
    >>> image = dataset["test"]["image"][0]

    >>> image_processor = AutoImageProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = image_processor(image, return_tensors="pt")

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> # model predicts one of the 1000 ImageNet classes
    >>> predicted_label = logits.argmax(-1).item()
    >>> print(model.config.id2label[predicted_label])
    {expected_output}
    ```
"""


PT_SAMPLE_DOCSTRINGS = {
    "SequenceClassification": PT_SEQUENCE_CLASSIFICATION_SAMPLE,
    "QuestionAnswering": PT_QUESTION_ANSWERING_SAMPLE,
    "TokenClassification": PT_TOKEN_CLASSIFICATION_SAMPLE,
    "MultipleChoice": PT_MULTIPLE_CHOICE_SAMPLE,
    "MaskedLM": PT_MASKED_LM_SAMPLE,
    "LMHead": PT_CAUSAL_LM_SAMPLE,
    "BaseModel": PT_BASE_MODEL_SAMPLE,
    "SpeechBaseModel": PT_SPEECH_BASE_MODEL_SAMPLE,
    "CTC": PT_SPEECH_CTC_SAMPLE,
    "AudioClassification": PT_SPEECH_SEQ_CLASS_SAMPLE,
    "AudioFrameClassification": PT_SPEECH_FRAME_CLASS_SAMPLE,
    "AudioXVector": PT_SPEECH_XVECTOR_SAMPLE,
    "VisionBaseModel": PT_VISION_BASE_MODEL_SAMPLE,
    "ImageClassification": PT_VISION_SEQ_CLASS_SAMPLE,
}


TF_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer(
    ...     "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="tf"
    ... )

    >>> logits = model(**inputs).logits
    >>> predicted_token_class_ids = tf.math.argmax(logits, axis=-1)

    >>> # Note that tokens are classified rather then input words which means that
    >>> # there might be more predicted token classes than words.
    >>> # Multiple token classes might account for the same word
    >>> predicted_tokens_classes = [model.config.id2label[t] for t in predicted_token_class_ids[0].numpy().tolist()]
    >>> predicted_tokens_classes
    {expected_output}
    ```

    ```python
    >>> labels = predicted_token_class_ids
    >>> loss = tf.math.reduce_mean(model(**inputs, labels=labels).loss)
    >>> round(float(loss), 2)
    {expected_loss}
    ```
"""

TF_QUESTION_ANSWERING_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

    >>> inputs = tokenizer(question, text, return_tensors="tf")
    >>> outputs = model(**inputs)

    >>> answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
    >>> answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])

    >>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    >>> tokenizer.decode(predict_answer_tokens)
    {expected_output}
    ```

    ```python
    >>> # target is "nice puppet"
    >>> target_start_index = tf.constant([{qa_target_start_index}])
    >>> target_end_index = tf.constant([{qa_target_end_index}])

    >>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
    >>> loss = tf.math.reduce_mean(outputs.loss)
    >>> round(float(loss), 2)
    {expected_loss}
    ```
"""

TF_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")

    >>> logits = model(**inputs).logits

    >>> predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
    >>> model.config.id2label[predicted_class_id]
    {expected_output}
    ```

    ```python
    >>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
    >>> num_labels = len(model.config.id2label)
    >>> model = {model_class}.from_pretrained("{checkpoint}", num_labels=num_labels)

    >>> labels = tf.constant(1)
    >>> loss = model(**inputs, labels=labels).loss
    >>> round(float(loss), 2)
    {expected_loss}
    ```
"""

TF_MASKED_LM_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors="tf")
    >>> logits = model(**inputs).logits

    >>> # retrieve index of {mask}
    >>> mask_token_index = tf.where((inputs.input_ids == tokenizer.mask_token_id)[0])
    >>> selected_logits = tf.gather_nd(logits[0], indices=mask_token_index)

    >>> predicted_token_id = tf.math.argmax(selected_logits, axis=-1)
    >>> tokenizer.decode(predicted_token_id)
    {expected_output}
    ```

    ```python
    >>> labels = tokenizer("The capital of France is Paris.", return_tensors="tf")["input_ids"]
    >>> # mask labels of non-{mask} tokens
    >>> labels = tf.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

    >>> outputs = model(**inputs, labels=labels)
    >>> round(float(outputs.loss), 2)
    {expected_loss}
    ```
"""

TF_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
    >>> outputs = model(inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

TF_MULTIPLE_CHOICE_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    >>> choice0 = "It is eaten with a fork and a knife."
    >>> choice1 = "It is eaten while held in the hand."

    >>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="tf", padding=True)
    >>> inputs = {{k: tf.expand_dims(v, 0) for k, v in encoding.items()}}
    >>> outputs = model(inputs)  # batch size is 1

    >>> # the linear classifier still needs to be trained
    >>> logits = outputs.logits
    ```
"""

TF_CAUSAL_LM_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
    >>> outputs = model(inputs)
    >>> logits = outputs.logits
    ```
"""

TF_SPEECH_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoProcessor, {model_class}
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> processor = AutoProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="tf")
    >>> outputs = model(**inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    >>> list(last_hidden_states.shape)
    {expected_output}
    ```
"""

TF_SPEECH_CTC_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoProcessor, {model_class}
    >>> from datasets import load_dataset
    >>> import tensorflow as tf

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> processor = AutoProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="tf")
    >>> logits = model(**inputs).logits
    >>> predicted_ids = tf.math.argmax(logits, axis=-1)

    >>> # transcribe speech
    >>> transcription = processor.batch_decode(predicted_ids)
    >>> transcription[0]
    {expected_output}
    ```

    ```python
    >>> inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="tf").input_ids

    >>> # compute loss
    >>> loss = model(**inputs).loss
    >>> round(float(loss), 2)
    {expected_loss}
    ```
"""

TF_VISION_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoImageProcessor, {model_class}
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("huggingface/cats-image")
    >>> image = dataset["test"]["image"][0]

    >>> image_processor = AutoImageProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = image_processor(image, return_tensors="tf")
    >>> outputs = model(**inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    >>> list(last_hidden_states.shape)
    {expected_output}
    ```
"""

TF_VISION_SEQ_CLASS_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoImageProcessor, {model_class}
    >>> import tensorflow as tf
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("huggingface/cats-image")
    >>> image = dataset["test"]["image"][0]

    >>> image_processor = AutoImageProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = image_processor(image, return_tensors="tf")
    >>> logits = model(**inputs).logits

    >>> # model predicts one of the 1000 ImageNet classes
    >>> predicted_label = int(tf.math.argmax(logits, axis=-1))
    >>> print(model.config.id2label[predicted_label])
    {expected_output}
    ```
"""

TF_SAMPLE_DOCSTRINGS = {
    "SequenceClassification": TF_SEQUENCE_CLASSIFICATION_SAMPLE,
    "QuestionAnswering": TF_QUESTION_ANSWERING_SAMPLE,
    "TokenClassification": TF_TOKEN_CLASSIFICATION_SAMPLE,
    "MultipleChoice": TF_MULTIPLE_CHOICE_SAMPLE,
    "MaskedLM": TF_MASKED_LM_SAMPLE,
    "LMHead": TF_CAUSAL_LM_SAMPLE,
    "BaseModel": TF_BASE_MODEL_SAMPLE,
    "SpeechBaseModel": TF_SPEECH_BASE_MODEL_SAMPLE,
    "CTC": TF_SPEECH_CTC_SAMPLE,
    "VisionBaseModel": TF_VISION_BASE_MODEL_SAMPLE,
    "ImageClassification": TF_VISION_SEQ_CLASS_SAMPLE,
}


FLAX_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    ```
"""

FLAX_QUESTION_ANSWERING_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    >>> inputs = tokenizer(question, text, return_tensors="jax")

    >>> outputs = model(**inputs)
    >>> start_scores = outputs.start_logits
    >>> end_scores = outputs.end_logits
    ```
"""

FLAX_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    ```
"""

FLAX_MASKED_LM_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors="jax")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    ```
"""

FLAX_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
    >>> outputs = model(**inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

FLAX_MULTIPLE_CHOICE_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    >>> choice0 = "It is eaten with a fork and a knife."
    >>> choice1 = "It is eaten while held in the hand."

    >>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="jax", padding=True)
    >>> outputs = model(**{{k: v[None, :] for k, v in encoding.items()}})

    >>> logits = outputs.logits
    ```
"""

FLAX_CAUSAL_LM_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
    >>> outputs = model(**inputs)

    >>> # retrieve logts for next token
    >>> next_token_logits = outputs.logits[:, -1]
    ```
"""

FLAX_SAMPLE_DOCSTRINGS = {
    "SequenceClassification": FLAX_SEQUENCE_CLASSIFICATION_SAMPLE,
    "QuestionAnswering": FLAX_QUESTION_ANSWERING_SAMPLE,
    "TokenClassification": FLAX_TOKEN_CLASSIFICATION_SAMPLE,
    "MultipleChoice": FLAX_MULTIPLE_CHOICE_SAMPLE,
    "MaskedLM": FLAX_MASKED_LM_SAMPLE,
    "BaseModel": FLAX_BASE_MODEL_SAMPLE,
    "LMHead": FLAX_CAUSAL_LM_SAMPLE,
}

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def filter_outputs_from_example(docstring, **kwargs):
    """
    Removes the lines testing an output with the doctest syntax in a code sample when it's set to `None`.
    """
    for key, value in kwargs.items():
        if value is not None:
            continue

        doc_key = "{" + key + "}"
        docstring = re.sub(rf"\n([^\n]+)\n\s+{doc_key}\n", "\n", docstring)

    return docstring

try:
    USE_TF = os.environ.get("USE_TF", "AUTO").upper()
    USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
    if USE_TORCH in ("1", "ON", "YES", "AUTO") and USE_TF not in ("1", "ON", "YES"):
        import torch

        _torch_available = True  # pylint: disable=invalid-name
        logger.info("PyTorch version {} available.".format(torch.__version__))
    else:
        logger.info("Disabling PyTorch because USE_TF is set")
        _torch_available = False
except ImportError:
    _torch_available = False  # pylint: disable=invalid-name

try:
    USE_TF = os.environ.get("USE_TF", "AUTO").upper()
    USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()

    if USE_TF in ("1", "ON", "YES", "AUTO") and USE_TORCH not in ("1", "ON", "YES"):
        import tensorflow as tf

        assert hasattr(tf, "__version__") and int(tf.__version__[0]) >= 2
        _tf_available = True  # pylint: disable=invalid-name
        logger.info("TensorFlow version {} available.".format(tf.__version__))
    else:
        logger.info("Disabling Tensorflow because USE_TORCH is set")
        _tf_available = False
except (ImportError, AssertionError):
    _tf_available = False  # pylint: disable=invalid-name


try:
    from torch.hub import _get_torch_home

    torch_cache_home = _get_torch_home()
except ImportError:
    torch_cache_home = os.path.expanduser(
        os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))
    )


try:
    import torch_xla.core.xla_model as xm  # noqa: F401

    if _torch_available:
        _torch_tpu_available = True  # pylint: disable=
    else:
        _torch_tpu_available = False
except ImportError:
    _torch_tpu_available = False


try:
    import psutil  # noqa: F401

    _psutil_available = True

except ImportError:
    _psutil_available = False


try:
    import py3nvml  # noqa: F401

    _py3nvml_available = True

except ImportError:
    _py3nvml_available = False


try:
    from apex import amp  # noqa: F401

    _has_apex = True
except ImportError:
    _has_apex = False

default_cache_path = os.path.join(torch_cache_home, "transformers")


PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", default_cache_path)
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)

WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF_WEIGHTS_NAME = "model.ckpt"
CONFIG_NAME = "config.json"
MODEL_CARD_NAME = "modelcard.json"


MULTIPLE_CHOICE_DUMMY_INPUTS = [[[0], [1]], [[0], [1]]]
DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
DUMMY_MASK = [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]

S3_BUCKET_PREFIX = "https://s3.amazonaws.com/models.huggingface.co/bert"
CLOUDFRONT_DISTRIB_PREFIX = "https://cdn.huggingface.co"


def is_torch_available():
    return _torch_available


def is_tf_available():
    return _tf_available


def is_torch_tpu_available():
    return _torch_tpu_available


def is_psutil_available():
    return _psutil_available


def is_py3nvml_available():
    return _py3nvml_available


def is_apex_available():
    return _has_apex


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def add_start_docstrings_to_callable(*docstr):
    def docstring_decorator(fn):
        class_name = ":class:`~transformers.{}`".format(fn.__qualname__.split(".")[0])
        intro = "   The {} forward method, overrides the :func:`__call__` special method.".format(class_name)
        note = r"""

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        pre and post processing steps while the latter silently ignores them.
        """
        fn.__doc__ = intro + note + "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def add_end_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = fn.__doc__ + "".join(docstr)
        return fn

    return docstring_decorator


RETURN_INTRODUCTION = r"""
    Returns:
        :class:`~{full_output_type}` or :obj:`tuple(torch.FloatTensor)` (if ``return_tuple=True`` is passed or when ``config.return_tuple=True``) comprising various elements depending on the configuration (:class:`~transformers.{config_class}`) and inputs:
"""


def _prepare_output_docstrings(output_type, config_class):
    """
    Prepares the return part of the docstring using `output_type`.
    """
    docstrings = output_type.__doc__

    # Remove the head of the docstring to keep the list of args only
    lines = docstrings.split("\n")
    i = 0
    while i < len(lines) and re.search(r"^\s*(Args|Parameters):\s*$", lines[i]) is None:
        i += 1
    if i < len(lines):
        docstrings = "\n".join(lines[(i + 1) :])

    # Add the return introduction
    full_output_type = f"{output_type.__module__}.{output_type.__name__}"
    intro = RETURN_INTRODUCTION.format(full_output_type=full_output_type, config_class=config_class)
    return intro + docstrings


PT_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1

        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

PT_QUESTION_ANSWERING_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> start_positions = torch.tensor([1])
        >>> end_positions = torch.tensor([3])

        >>> outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
        >>> loss = outputs.loss
        >>> start_scores = outputs.start_scores
        >>> end_scores = outputs.end_scores
"""

PT_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

PT_MASKED_LM_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> input_ids = tokenizer("Hello, my dog is cute", return_tensors="pt")["input_ids"]

        >>> outputs = model(input_ids, labels=input_ids)
        >>> loss = outputs.loss
        >>> prediction_logits = outputs.logits
"""

PT_BASE_MODEL_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
"""

PT_MULTIPLE_CHOICE_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> choice0 = "It is eaten with a fork and a knife."
        >>> choice1 = "It is eaten while held in the hand."
        >>> labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

        >>> encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors='pt', padding=True)
        >>> outputs = model(**{{k: v.unsqueeze(0) for k,v in encoding.items()}}, labels=labels)  # batch size is 1

        >>> # the linear classifier still needs to be trained
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

PT_CAUSAL_LM_SAMPLE = r"""
    Example::

        >>> import torch
        >>> from transformers import {tokenizer_class}, {model_class}

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs, labels=inputs["input_ids"])
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

TF_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> input_ids = inputs["input_ids"]
        >>> inputs["labels"] = tf.reshape(tf.constant([1] * tf.size(input_ids).numpy()), (-1, tf.size(input_ids))) # Batch size 1

        >>> outputs = model(inputs)
        >>> loss, scores = outputs[:2]
"""

TF_QUESTION_ANSWERING_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        >>> input_dict = tokenizer(question, text, return_tensors='tf')
        >>> start_scores, end_scores = model(input_dict)

        >>> all_tokens = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
        >>> answer = ' '.join(all_tokens[tf.math.argmax(start_scores, 1)[0] : tf.math.argmax(end_scores, 1)[0]+1])
"""

TF_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> inputs["labels"] = tf.reshape(tf.constant(1), (-1, 1)) # Batch size 1

        >>> outputs = model(inputs)
        >>> loss, logits = outputs[:2]
"""

TF_MASKED_LM_SAMPLE = r"""
    Example::
        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]  # Batch size 1

        >>> outputs = model(input_ids)
        >>> prediction_scores = outputs[0]
"""

TF_BASE_MODEL_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> outputs = model(inputs)

        >>> last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
"""

TF_MULTIPLE_CHOICE_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> choice0 = "It is eaten with a fork and a knife."
        >>> choice1 = "It is eaten while held in the hand."

        >>> encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors='tf', padding=True)
        >>> inputs = {{k: tf.expand_dims(v, 0) for k, v in encoding.items()}}
        >>> outputs = model(inputs)  # batch size is 1

        >>> # the linear classifier still needs to be trained
        >>> logits = outputs[0]
"""

TF_CAUSAL_LM_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> outputs = model(inputs)
        >>> logits = outputs[0]
"""


def add_code_sample_docstrings(
    *docstr,
    processor_class=None,
    checkpoint=None,
    output_type=None,
    config_class=None,
    mask="[MASK]",
    qa_target_start_index=14,
    qa_target_end_index=15,
    model_cls=None,
    modality=None,
    expected_output=None,
    expected_loss=None,
    real_checkpoint=None,
):
    def docstring_decorator(fn):
        # model_class defaults to function's class if not specified otherwise
        model_class = fn.__qualname__.split(".")[0] if model_cls is None else model_cls

        if model_class[:2] == "TF":
            sample_docstrings = TF_SAMPLE_DOCSTRINGS
        elif model_class[:4] == "Flax":
            sample_docstrings = FLAX_SAMPLE_DOCSTRINGS
        else:
            sample_docstrings = PT_SAMPLE_DOCSTRINGS

        # putting all kwargs for docstrings in a dict to be used
        # with the `.format(**doc_kwargs)`. Note that string might
        # be formatted with non-existing keys, which is fine.
        doc_kwargs = {
            "model_class": model_class,
            "processor_class": processor_class,
            "checkpoint": checkpoint,
            "mask": mask,
            "qa_target_start_index": qa_target_start_index,
            "qa_target_end_index": qa_target_end_index,
            "expected_output": expected_output,
            "expected_loss": expected_loss,
            "real_checkpoint": real_checkpoint,
            "fake_checkpoint": checkpoint,
            "true": "{true}",  # For <Tip warning={true}> syntax that conflicts with formatting.
        }

        if ("SequenceClassification" in model_class or "AudioClassification" in model_class) and modality == "audio":
            code_sample = sample_docstrings["AudioClassification"]
        elif "SequenceClassification" in model_class:
            code_sample = sample_docstrings["SequenceClassification"]
        elif "QuestionAnswering" in model_class:
            code_sample = sample_docstrings["QuestionAnswering"]
        elif "TokenClassification" in model_class:
            code_sample = sample_docstrings["TokenClassification"]
        elif "MultipleChoice" in model_class:
            code_sample = sample_docstrings["MultipleChoice"]
        elif "MaskedLM" in model_class or model_class in ["FlaubertWithLMHeadModel", "XLMWithLMHeadModel"]:
            code_sample = sample_docstrings["MaskedLM"]
        elif "LMHead" in model_class or "CausalLM" in model_class:
            code_sample = sample_docstrings["LMHead"]
        elif "CTC" in model_class:
            code_sample = sample_docstrings["CTC"]
        elif "AudioFrameClassification" in model_class:
            code_sample = sample_docstrings["AudioFrameClassification"]
        elif "XVector" in model_class and modality == "audio":
            code_sample = sample_docstrings["AudioXVector"]
        elif "Model" in model_class and modality == "audio":
            code_sample = sample_docstrings["SpeechBaseModel"]
        elif "Model" in model_class and modality == "vision":
            code_sample = sample_docstrings["VisionBaseModel"]
        elif "Model" in model_class or "Encoder" in model_class:
            code_sample = sample_docstrings["BaseModel"]
        elif "ImageClassification" in model_class:
            code_sample = sample_docstrings["ImageClassification"]
        else:
            raise ValueError(f"Docstring can't be built for model {model_class}")

        code_sample = filter_outputs_from_example(
            code_sample, expected_output=expected_output, expected_loss=expected_loss
        )
        if real_checkpoint is not None:
            code_sample = FAKE_MODEL_DISCLAIMER + code_sample
        func_doc = (fn.__doc__ or "") + "".join(docstr)
        output_doc = "" if output_type is None else _prepare_output_docstrings(output_type, config_class)
        built_doc = code_sample.format(**doc_kwargs)
        fn.__doc__ = func_doc + output_doc + built_doc
        return fn

    return docstring_decorator


def replace_return_docstrings(output_type=None, config_class=None):
    def docstring_decorator(fn):
        docstrings = fn.__doc__
        lines = docstrings.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^\s*Returns?:\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            lines[i] = _prepare_output_docstrings(output_type, config_class)
            docstrings = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'Return:' or 'Returns:' in its docstring as placeholder, current docstring is:\n{docstrings}"
            )
        fn.__doc__ = docstrings
        return fn

    return docstring_decorator


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def hf_bucket_url(model_id: str, filename: str, use_cdn=True) -> str:
    """
    Resolve a model identifier, and a file name, to a HF-hosted url
    on either S3 or Cloudfront (a Content Delivery Network, or CDN).

    Cloudfront is replicated over the globe so downloads are way faster
    for the end user (and it also lowers our bandwidth costs). However, it
    is more aggressively cached by default, so may not always reflect the
    latest changes to the underlying file (default TTL is 24 hours).

    In terms of client-side caching from this library, even though
    Cloudfront relays the ETags from S3, using one or the other
    (or switching from one to the other) will affect caching: cached files
    are not shared between the two because the cached file's name contains
    a hash of the url.
    """
    endpoint = CLOUDFRONT_DISTRIB_PREFIX if use_cdn else S3_BUCKET_PREFIX
    legacy_format = "/" not in model_id
    if legacy_format:
        return f"{endpoint}/{model_id}-{filename}"
    else:
        return f"{endpoint}/{model_id}/{filename}"


def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    If the url ends with .h5 (Keras HDF5 weights) adds '.h5' to the name
    so that TF 2.0 can identify it as a HDF5 file
    (see https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1380)
    """
    url_bytes = url.encode("utf-8")
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    if url.endswith(".h5"):
        filename += ".h5"

    return filename


def filename_to_url(filename, cache_dir=None):
    """
    Return the url and etag (which may be ``None``) stored for `filename`.
    Raise ``EnvironmentError`` if `filename` or its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise EnvironmentError("file {} not found".format(cache_path))

    meta_path = cache_path + ".json"
    if not os.path.exists(meta_path):
        raise EnvironmentError("file {} not found".format(meta_path))

    with open(meta_path, encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)
    url = metadata["url"]
    etag = metadata["etag"]

    return url, etag


def cached_path(
    url_or_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    extract_compressed_file=False,
    force_extract=False,
    local_files_only=False,
) -> Optional[str]:
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    Args:
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-dowload the file even if it's already cached in the cache dir.
        resume_download: if True, resume the download if incompletly recieved file is found.
        user_agent: Optional string or dict that will be appended to the user-agent on remote requests.
        extract_compressed_file: if True and the path point to a zip or tar file, extract the compressed
            file in a folder along the archive.
        force_extract: if True when extract_compressed_file is True and the archive was already extracted,
            re-extract the archive and overide the folder where it was extracted.

    Return:
        None in case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
        Local path (string) otherwise
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if is_remote_url(url_or_filename):
        # URL, so get it from the cache (downloading if necessary)
        output_path = get_from_cache(
            url_or_filename,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            user_agent=user_agent,
            local_files_only=local_files_only,
        )
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == "":
        # File, but it doesn't exist.
        raise EnvironmentError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))

    if extract_compressed_file:
        if not is_zipfile(output_path) and not tarfile.is_tarfile(output_path):
            return output_path

        # Path where we extract compressed archives
        # We avoid '.' in dir name and add "-extracted" at the end: "./model.zip" => "./model-zip-extracted/"
        output_dir, output_file = os.path.split(output_path)
        output_extract_dir_name = output_file.replace(".", "-") + "-extracted"
        output_path_extracted = os.path.join(output_dir, output_extract_dir_name)

        if os.path.isdir(output_path_extracted) and os.listdir(output_path_extracted) and not force_extract:
            return output_path_extracted

        # Prevent parallel extractions
        lock_path = output_path + ".lock"
        with FileLock(lock_path):
            shutil.rmtree(output_path_extracted, ignore_errors=True)
            os.makedirs(output_path_extracted)
            if is_zipfile(output_path):
                with ZipFile(output_path, "r") as zip_file:
                    zip_file.extractall(output_path_extracted)
                    zip_file.close()
            elif tarfile.is_tarfile(output_path):
                tar_file = tarfile.open(output_path)
                tar_file.extractall(output_path_extracted)
                tar_file.close()
            else:
                raise EnvironmentError("Archive format of {} could not be identified".format(output_path))

        return output_path_extracted

    return output_path


def http_get(url, temp_file, proxies=None, resume_size=0, user_agent: Union[Dict, str, None] = None):
    ua = "transformers/{}; python/{}".format(__version__, sys.version.split()[0])
    if is_torch_available():
        ua += "; torch/{}".format(torch.__version__)
    if is_tf_available():
        ua += "; tensorflow/{}".format(tf.__version__)
    if isinstance(user_agent, dict):
        ua += "; " + "; ".join("{}/{}".format(k, v) for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    headers = {"user-agent": ua}
    if resume_size > 0:
        headers["Range"] = "bytes=%d-" % (resume_size,)
    response = requests.get(url, stream=True, proxies=proxies, headers=headers)
    if response.status_code == 416:  # Range not satisfiable
        return
    content_length = response.headers.get("Content-Length")
    total = resume_size + int(content_length) if content_length is not None else None
    progress = tqdm(
        unit="B",
        unit_scale=True,
        total=total,
        initial=resume_size,
        desc="Downloading",
        disable=bool(logger.getEffectiveLevel() == logging.NOTSET),
    )
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def get_from_cache(
    url,
    cache_dir=None,
    force_download=False,
    proxies=None,
    etag_timeout=10,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    local_files_only=False,
) -> Optional[str]:
    """
    Given a URL, look for the corresponding file in the local cache.
    If it's not there, download it. Then return the path to the cached file.

    Return:
        None in case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
        Local path (string) otherwise
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    etag = None
    if not local_files_only:
        try:
            response = requests.head(url, allow_redirects=True, proxies=proxies, timeout=etag_timeout)
            if response.status_code == 200:
                etag = response.headers.get("ETag")
        except (EnvironmentError, requests.exceptions.Timeout):
            # etag is already None
            pass

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    # etag is None = we don't have a connection, or url doesn't exist, or is otherwise inaccessible.
    # try to get the last downloaded one
    if etag is None:
        if os.path.exists(cache_path):
            return cache_path
        else:
            matching_files = [
                file
                for file in fnmatch.filter(os.listdir(cache_dir), filename + ".*")
                if not file.endswith(".json") and not file.endswith(".lock")
            ]
            if len(matching_files) > 0:
                return os.path.join(cache_dir, matching_files[-1])
            else:
                # If files cannot be found and local_files_only=True,
                # the models might've been found if local_files_only=False
                # Notify the user about that
                if local_files_only:
                    raise ValueError(
                        "Cannot find the requested files in the cached path and outgoing traffic has been"
                        " disabled. To enable model look-ups and downloads online, set 'local_files_only'"
                        " to False."
                    )
                return None

    # From now on, etag is not None.
    if os.path.exists(cache_path) and not force_download:
        return cache_path

    # Prevent parallel downloads of the same file with a lock.
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):

        # If the download just completed while the lock was activated.
        if os.path.exists(cache_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            return cache_path

        if resume_download:
            incomplete_path = cache_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager():
                with open(incomplete_path, "a+b") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(tempfile.NamedTemporaryFile, dir=cache_dir, delete=False)
            resume_size = 0

        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            logger.info("%s not found in cache or force_download set to True, downloading to %s", url, temp_file.name)

            http_get(url, temp_file, proxies=proxies, resume_size=resume_size, user_agent=user_agent)

        logger.info("storing %s in cache at %s", url, cache_path)
        os.replace(temp_file.name, cache_path)

        logger.info("creating metadata file for %s", cache_path)
        meta = {"url": url, "etag": etag}
        meta_path = cache_path + ".json"
        with open(meta_path, "w") as meta_file:
            json.dump(meta, meta_file)

    return cache_path


class cached_property(property):
    """
    Descriptor that mimics @property but caches output in member variable.

    From tensorflow_datasets

    Built-in in functools from Python 3.8.
    """

    def __get__(self, obj, objtype=None):
        # See docs.python.org/3/howto/descriptor.html#properties
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        attr = "__cached_" + self.fget.__name__
        cached = getattr(obj, attr, None)
        if cached is None:
            cached = self.fget(obj)
            setattr(obj, attr, cached)
        return cached


def torch_required(func):
    # Chose a different decorator name than in tests so it's clear they are not the same.
    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_torch_available():
            return func(*args, **kwargs)
        else:
            raise ImportError(f"Method `{func.__name__}` requires PyTorch.")

    return wrapper


def tf_required(func):
    # Chose a different decorator name than in tests so it's clear they are not the same.
    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_tf_available():
            return func(*args, **kwargs)
        else:
            raise ImportError(f"Method `{func.__name__}` requires TF.")

    return wrapper


class ModelOutput:
    """
    Base class for all model outputs as dataclass. Has a ``__getitem__`` that allows indexing by integer or slice (like
    a tuple) or strings (like a dictionnary) that will ignore the ``None`` attributes.
    """

    def to_tuple(self):
        """
        Converts :obj:`self` to a tuple.

        Return: A tuple containing all non-:obj:`None` attributes of the :obj:`self`.
        """
        return tuple(getattr(self, f) for f in self.__dataclass_fields__.keys() if getattr(self, f, None) is not None)

    def to_dict(self):
        """
        Converts :obj:`self` to a Python dictionary.

        Return: A dictionary containing all non-:obj:`None` attributes of the :obj:`self`.
        """
        return {f: getattr(self, f) for f in self.__dataclass_fields__.keys() if getattr(self, f, None) is not None}

    def __getitem__(self, i):
        return self.to_dict()[i] if isinstance(i, str) else self.to_tuple()[i]

    def __len__(self):
        return len(self.to_tuple())

def add_start_docstrings_to_model_forward(*docstr):
    def docstring_decorator(fn):
        docstring = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        class_name = f"[`{fn.__qualname__.split('.')[0]}`]"
        intro = f"   The {class_name} forward method, overrides the `__call__` special method."
        note = r"""

    <Tip>

    Although the recipe for forward pass needs to be defined within this function, one should call the [`Module`]
    instance afterwards instead of this since the former takes care of running the pre and post processing steps while
    the latter silently ignores them.

    </Tip>
"""

        fn.__doc__ = intro + note + docstring
        return fn

    return docstring_decorator
