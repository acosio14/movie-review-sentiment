from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from transformers import DistilBertConfig, DataCollatorWithPadding, TrainingArguments, Trainer
import evaluate
import numpy as np

rotten_tomatoes = load_dataset("rotten_tomatoes")

imbd_dataset = load_dataset('csv', data_files='IMBD_Dataset.csv')
imbd = imbd_dataset['train'].train_test_split(test_size=0.2)
label_map = {
    "negative": 0,
    "positive": 1,
}
def encode_labels(dataset):
    dataset["text"] = label_map[dataset["review"]]
    dataset["labels"] = label_map[dataset["sentiment"]]
    return dataset

imbd = imbd.map(encode_labels).remove_columns(['review','sentimnet'])

model_name = "distilbert/distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.config = DistilBertConfig.from_pretrained(
    model_name,
    id2label={0: "Negative", 1:"Positive"},
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_data(dataset):
    return tokenizer(
        dataset["review"],
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

imbd = imbd_dataset.map(tokenize_data, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="distilbert-rotten-tomatoes",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    dataloader_pin_memory=False,
    label_names=["sentiment"],
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=imbd["train"],
    eval_dataset=imbd["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()