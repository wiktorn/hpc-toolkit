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


def main():
    """Run LLM finetuning"""

    # model_name = "meta-llama/Llama-3.1-70B-Instruct"
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    parser = HfArgumentParser(TrainingArguments)
    (training_args,) = parser.parse_args_into_dataclasses()

    # Prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("databricks/databricks-dolly-15k")
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
        item = tokenizer(item)
        return item

    dataset = dataset.map(format_chat, remove_columns=column_names)

    # Do training
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
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
