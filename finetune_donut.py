# train_donut_fixed.py
import json
from datasets import load_dataset
from PIL import Image
import numpy as np
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments

model_id = "naver-clova-ix/donut-base-finetuned-cord-v2"

# load processor + model
processor = DonutProcessor.from_pretrained(model_id, use_fast=False, image_size=[600, 400])
model = VisionEncoderDecoderModel.from_pretrained(model_id, use_safetensors=True)

# ensure tokenizer pad/eos/bos exist and are consistent
# We'll register both opening and closing custom tags as BOS/EOS special tokens.
# Also we will avoid adding thousands of long tokens blindly; keep keys short if possible.
processor.tokenizer.add_special_tokens({
    "bos_token": "<s_custom>",
    "eos_token": "</s_custom>"
})
# If you want to add domain-specific keys, add only short token forms (recommended):
# e.g. make mapping {"po_number":"<po_number>", "customer_info":"<customer_info>", ...}
# then add them as additional_special_tokens. Below is the minimal approach:
# processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<po_number>", "<customer_info>", ...]})

# resize embeddings for whole model (safer)
model.decoder.resize_token_embeddings(len(processor.tokenizer))

# set model config tokens
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

# load json dataset
raw_dataset = load_dataset("json", data_files={"train":"train3.json", "validation":"val3.json"})
print("Columns in dataset:", raw_dataset["train"].column_names)

# helper: convert nested ground-truth dict to a JSON string
def gt_to_text(gt_dict):
    return json.dumps(gt_dict, ensure_ascii=False)

# Preprocess: return numpy arrays / lists (not torch tensors)
def preprocess(example):
    image = Image.open(example["image"]).convert("RGB")
    # return_tensors='np' to store numpy arrays in the dataset (safer for mapping)
    px = processor(image, return_tensors="np").pixel_values[0]  # shape (C,H,W)
    text = gt_to_text(example["ground_truth"])
    # wrap with our special BOS/EOS tokens (tokenizer knows them)
    text_with_tags = f"{processor.tokenizer.bos_token}{text}{processor.tokenizer.eos_token}"

    tokenized = processor.tokenizer(
        text_with_tags,
        add_special_tokens=False,  # we already manually added BOS/EOS in the string
        truncation=True,
        max_length=512
    )
    labels = tokenized["input_ids"]  # plain python list

    return {"pixel_values": px, "labels": labels}

processed_dataset = raw_dataset.map(preprocess, remove_columns=raw_dataset["train"].column_names)

print("Final dataset columns:", processed_dataset["train"].column_names)

# Data collator: pads labels with -100 so loss ignores padding
def donut_data_collator(features):
    # pixel_values -> stack into tensor BxCxHxW
    pixel_values = torch.stack([torch.tensor(f["pixel_values"]) for f in features])

    # pad labels to max length in batch with pad_token_id then convert pads to -100
    label_lists = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
    padded_labels = torch.nn.utils.rnn.pad_sequence(label_lists, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    padded_labels[padded_labels == processor.tokenizer.pad_token_id] = -100

    return {"pixel_values": pixel_values, "labels": padded_labels}

# Training args (keep reasonable)
training_args = Seq2SeqTrainingArguments(
    output_dir="./donut-finetuned",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    warmup_steps=100,
    num_train_epochs=50,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    report_to="none",
    eval_strategy="steps",
    gradient_checkpointing=True,
    optim="adafactor",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    data_collator=donut_data_collator,
    tokenizer=processor.tokenizer,   # pass tokenizer so trainer can handle generation/prediction
)

print("Train steps (dataloader len):", len(trainer.get_train_dataloader()))

if torch.cuda.is_available():
    torch.cuda.empty_cache()

trainer.train()

# save artifacts
model.save_pretrained("./donut-finetuned")
processor.save_pretrained("./donut-finetuned")
