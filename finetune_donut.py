from transformers import DonutProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
from PIL import Image
import torch

# -----------------------------
# Step 1: Load model and processor
# -----------------------------
model_id = "naver-clova-ix/donut-base-finetuned-cord-v2"

processor = DonutProcessor.from_pretrained(
    model_id, 
    use_fast=False, 
    image_size=[600, 400]
)
model = VisionEncoderDecoderModel.from_pretrained(model_id, use_safetensors=True)

# Ensure tokenizer has pad/eos setup
processor.tokenizer.pad_token = processor.tokenizer.eos_token
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s>")

# -----------------------------
# Step 2: Load dataset from JSONL
# -----------------------------
raw_dataset = load_dataset(
    "json",
    data_files={
        "train": "train.jsonl",
        "validation": "val.jsonl"
    }
)
print("Columns in dataset:", raw_dataset["train"].column_names)

# -----------------------------
# Step 3: Preprocessing function
# -----------------------------
def preprocess(example):
    image = Image.open(example["image"]).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.squeeze(0)

    # ✅ FIX: dataset already has <s_custom> ... </s_custom>, so just clean it
    text = example["ground_truth"].strip()

    # normalize: if missing start/end tags, add them
    if not text.startswith("<s_custom>"):
        text = f"<s_custom>{text}"
    if not text.endswith("</s_custom>"):
        text = f"{text}</s_custom>"

    labels = processor.tokenizer(
        text,
        truncation=True,
        max_length=512,   # allow longer JSON
        return_tensors="pt"
    ).input_ids.squeeze(0)

    return {"pixel_values": pixel_values, "labels": labels}

# ✅ FIX: actually preprocess the dataset
processed_dataset = raw_dataset.map(
    preprocess,
    remove_columns=raw_dataset["train"].column_names
)
print("Final dataset columns:", processed_dataset["train"].column_names)

# -----------------------------
# Step 4: Data collator
# -----------------------------
def donut_data_collator(features):
    pixel_values = torch.stack([torch.tensor(f["pixel_values"]) for f in features])
    labels = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(f["labels"]) for f in features],
        batch_first=True,
        padding_value=-100
    )
    return {"pixel_values": pixel_values, "labels": labels}

# -----------------------------
# Step 5: Training arguments
# -----------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="./donut-finetuned",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    warmup_steps=500,
    num_train_epochs=30,
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

# -----------------------------
# Step 6: Trainer
# -----------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    data_collator=donut_data_collator,
    processing_class=processor
)

print("Train steps:", len(trainer.get_train_dataloader()))

# -----------------------------
# Step 7: Train
# -----------------------------
if torch.cuda.is_available():
    torch.cuda.empty_cache()

trainer.train()

# -----------------------------
# Step 8: Save model + processor
# -----------------------------
model.save_pretrained("./donut-finetuned")
processor.save_pretrained("./donut-finetuned")
