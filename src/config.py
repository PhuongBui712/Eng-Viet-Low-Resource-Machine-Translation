import torch


class BaseConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs:
            setattr(self, k, v)


class NMTConfig(BaseConfig):
    # data
    src_lang = 'en'
    tgt_lang = 'vi'
    max_length = 128
    add_special_tokens = True
    cache_dir = '../.cache'
    augmented_data_size = 0.0001

    # model
    model_name = "Helsinki-NLP/opus-mt-en-vi"


    # training arguments
    device = 'cuda' if torch.cuda.is_available() else \
        ('mps' if torch.backends.mps.is_available() else 'cpu')

    use_mps_device = False
    if device == 'mps':
        use_mps_device = True

    learning_rate = 5e-5
    train_batch_size = 16
    eval_batch_size = 16
    num_train_epochs = 2
    save_total_limit = 1
    ckpt_dir = '../.cache/checkpoints'
    eval_steps = 1000

    # inference
    beam_size = 5

