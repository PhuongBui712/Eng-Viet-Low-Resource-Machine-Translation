from config import NMTConfig
from data_setup import NMTDataset

import numpy as np
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


if __name__ == '__main__':
    cfg = NMTConfig()
    cfg.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, cache_dir=cfg.cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name, cache_dir=cfg.cache_dir)

    train_dataset = train_dataset = NMTDataset(cfg, 'train')
    valid_dataset = NMTDataset(cfg, 'validation')
    test_dataset = NMTDataset(cfg, 'test')

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
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_evaluate_metric
    )

    trainer.train()
    trainer.save_model('./models/EnViT5')