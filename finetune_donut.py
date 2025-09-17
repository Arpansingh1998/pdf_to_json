from transformers import DonutProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
from PIL import Image
import torch

# -----------------------------
# Step 1: Load model and processor
# -----------------------------
model_id = "naver-clova-ix/donut-base"

# Processor + Model
processor = DonutProcessor.from_pretrained(model_id, use_fast=False)
print("111111111111111111111111111111111111111111111111111")
print(processor)
model = VisionEncoderDecoderModel.from_pretrained(model_id)
print("22222222222222222222222222222222222222222222222222")
print(model)

# Fix tokenizer tokens
processor.tokenizer.pad_token = processor.tokenizer.eos_token
print("33333333333333333333333333333333333333333333333")
print(processor.tokenizer.pad_token)
model.config.pad_token_id = processor.tokenizer.pad_token_id
print("44444444444444444444444444444444444444444444444")
print(model.config.pad_token_id)
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s>")
print("55555555555555555555555555555555555555555555555")
print(model.config.decoder_start_token_id)

# -----------------------------
# Step 2: Load dataset
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
# Step 3: Preprocess
# -----------------------------
def preprocess(example):
    image = Image.open(example["image"]).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.squeeze(0)

    labels = processor.tokenizer(
        example["ground_truth"],
        truncation=True,
        max_length=256,   # keep short for memory
        return_tensors="pt"
    ).input_ids.squeeze(0)

    labels[labels == processor.tokenizer.pad_token_id] = -100
    print("666666666666666666666666666666666666666666666666")
    print({"pixel_values": pixel_values, "labels": labels})
    return {"pixel_values": pixel_values, "labels": labels}

processed_dataset = raw_dataset.map(
    preprocess,
    remove_columns=raw_dataset["train"].column_names
)
print("Final dataset columns:", processed_dataset["train"].column_names)

print("77777777777777777777777777777777777777777777777777777")
print(processed_dataset)
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
    print("888888888888888888888888888888888888888888888888888888")
    print({"pixel_values": pixel_values, "labels": labels})
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
    warmup_steps=50,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="epoch",   # ✅ ensures saving even on small dataset
    save_total_limit=2,
    fp16=False,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    report_to="none",
    gradient_checkpointing=True,
    optim="adafactor",
)

print("9999999999999999999999999999999999999999999999999999999999999999999")
print(training_args)
# -----------------------------
# Step 6: Trainer
# -----------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    data_collator=donut_data_collator,
    tokenizer=processor.tokenizer
)

print("Train steps (batches per epoch):", len(trainer.get_train_dataloader()))

print("10101010101010101010101010101010101010101010101010101010101010101010")
print(trainer)
# -----------------------------
# Step 7: Train
# -----------------------------
if torch.cuda.is_available():
    torch.cuda.empty_cache()

trainer.train()

print("Training completed!")
print("121212121212121212121212121212121212121212121212121212121212121212")
# -----------------------------
# Step 8: Save model + processor
# -----------------------------
trainer.save_model("./donut-finetuned")         # ✅ always saves model
processor.save_pretrained("./donut-finetuned")  # ✅ saves tokenizer/processor

# -----------------------------
# Step 9: Test inference
# -----------------------------
print("\nRunning test inference...")
sample = raw_dataset["validation"][0]
image = Image.open(sample["image"]).convert("RGB")

inputs = processor(image, return_tensors="pt").pixel_values
outputs = model.generate(inputs, max_length=256)

pred = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print("Predicted:", pred)
print("Ground truth:", sample["ground_truth"])
