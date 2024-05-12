from config import NMTConfig
from data_setup import NMTDataset, Augmented_NMTDataset
from inference import inference
from utils import load_data

import os
import numpy as np
from tqdm.autonotebook import tqdm
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels


def batch_translation(texts, batch_size=32):
    res = []
    for i in tqdm(range(len(texts) // batch_size + 1)):
        batch_data = texts[i * batch_size: min((i + 1) * batch_size, len(texts))]
        output_strs = inference(batch_data, cfg.tokenizer, model)
        output_strs = [output.strip('en: ') for output in output_strs]
        res += output_strs

    return res

def compute_metrics_with_tokenizer(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        metric = evaluate.load('sacrebleu')
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}

        return result

    return compute_metrics


def train(model, cfg, train_dataset, eval_dataset):
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.ckpt_dir,
        predict_with_generate=True,
        evaluation_strategy='steps',
        save_strategy='steps',
        eval_steps=cfg.eval_steps,
        save_steps=cfg.eval_steps,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        use_mps_device=cfg.use_mps_device,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=cfg.tokenizer,
        model=model
    )

    compute_evaluate_metric = compute_metrics_with_tokenizer(cfg.tokenizer)

    trainer = Seq2SeqTrainer(
        tokenizer=cfg.tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_evaluate_metric
    )

    trainer.train()

    return trainer


if __name__ == '__main__':
    cfg = NMTConfig()
    cfg.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, cache_dir=cfg.cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name, cache_dir=cfg.cache_dir)

    model_dir = '../models/finetuned-EnViT5'
    if not os.path.isdir(model_dir):
        train_dataset = NMTDataset(cfg, 'train')
        valid_dataset = NMTDataset(cfg, 'validation')

        trainer = train(model, cfg, train_dataset, valid_dataset)
        trainer.save_model(model_dir)

    augmented_model_dir = '../models/augmented-EnViT5'
    if not os.path.isdir(augmented_model_dir):
        # create synthetic data
        train_augmented_tgt_texts = load_data('../data/PhoMT/tokenization/train/train.vi')
        train_augmented_tgt_texts = train_augmented_tgt_texts[:int(cfg.augmented_data_size * len(train_augmented_tgt_texts))]

        eval_augmented_tgt_texts = load_data('../data/PhoMT/tokenization/dev/dev.vi')
        eval_augmented_tgt_texts = eval_augmented_tgt_texts[:int(cfg.augmented_data_size * len(eval_augmented_tgt_texts))]

        train_augmented_src_texts = batch_translation(train_augmented_tgt_texts)
        eval_augmented_src_texts = batch_translation(eval_augmented_tgt_texts)

        # aggregate into augmented data
        train_augmented_dataset = Augmented_NMTDataset(cfg,
                                                       augmented_src=train_augmented_src_texts,
                                                       augmented_tgt=train_augmented_tgt_texts,
                                                       split='train')
        eval_augmented_dataset = Augmented_NMTDataset(cfg,
                                                      augmented_src=eval_augmented_src_texts,
                                                      augmented_tgt=eval_augmented_tgt_texts,
                                                      split='validation')

        trainer = train(model, cfg, train_augmented_dataset, eval_augmented_dataset)
        trainer.save_model(model_dir)