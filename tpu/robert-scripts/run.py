from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)


@dataclass
class ModelArguments:
    model_name: str = field(
        default="meta-llama/Llama-3.1-70B-Instruct",
        metadata={"help": "Name or path to model to train"},
    )


@dataclass
class DatasetArguments:
    dataset_name: str = field(
        default="databricks/databricks-dolly-15k",
        metadata={"help": "Name or path to dataset to use"},
    )
    max_seq_len: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length to use from the dataset"},
    )


def main():
    """Run LLM finetuning"""

    parser = HfArgumentParser((ModelArguments, DatasetArguments, TrainingArguments))
    (model_args, dataset_args, training_args) = parser.parse_args_into_dataclasses()

    # Prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset(dataset_args.dataset_name)
    dataset = dataset["train"]
    column_names = dataset.column_names

    def format_chat(item):
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": item["context"] + item["instruction"]},
            {"role": "assistant", "content": item["response"]},
        ]
        item = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        item = tokenizer(item, return_length=True)
        return item

    dataset = dataset.map(format_chat, remove_columns=column_names)
    dataset = dataset.filter(lambda item: item["length"][0] <= dataset_args.max_seq_len)

    # Do training
    config = AutoConfig.from_pretrained(model_args.model_name)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name, config=config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
    )
    trainer.train()


if __name__ == "__main__":
    main()