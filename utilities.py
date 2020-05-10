import numpy as np
import os
import json

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences


def prepare_ans_ques_ref(ans, ques, ref):

    max_length = 40
    trunc_type = 'post'

    with open(os.path.join('data', 'tokenizer.json')) as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)

    ans_sequences = tokenizer.texts_to_sequences([ans])
    ques_sequences = tokenizer.texts_to_sequences([ques])
    new_list = ' '.join(ref)
    ref_sequences = tokenizer.texts_to_sequences([new_list])

    ans_padded = pad_sequences(ans_sequences, maxlen=max_length, truncating=trunc_type)
    ques_padded = pad_sequences(ques_sequences, maxlen=max_length, truncating=trunc_type)
    ref_padded = pad_sequences(ref_sequences, maxlen=2 * max_length, truncating=trunc_type)

    return [np.expand_dims(np.concatenate((ans_padded[0], ques_padded[0]), axis=0), axis=0), np.expand_dims(ref_padded[0], axis=0)]


def call_class(output):

    classes_dict = {
        0: 'contradictory',
        1: 'correct',
        2: 'correct but incomplete',
        3: 'incorrect'
    }

    return classes_dict[np.argmax(output)]
