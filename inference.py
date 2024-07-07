from src.config import NMTConfig

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel


def inference(text,
              tokenizer: PreTrainedTokenizer,
              model: PreTrainedModel,
              device=NMTConfig.device,
              max_length=NMTConfig.max_length,
              beam_size=NMTConfig.beam_size):
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    device = torch.device(device)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    model.to(device)

    outputs = model.generate(input_ids,
                             attention_mask=attention_mask,
                             max_length=max_length,
                             early_stopping=True,
                             num_beams=beam_size,
                             length_penalty=2.0)

    output_strs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    del input_ids
    del attention_mask

    return output_strs