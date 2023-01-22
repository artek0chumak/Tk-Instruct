import os
import torch
import json
import tqdm
import random
from transformers import HfArgumentParser, pipeline, AutoTokenizer, AutoModelForCausalLM
from run_s2s import DataTrainingArguments
from datasets import load_dataset
from dataclasses import dataclass, field
from nltk import sent_tokenize

from ni_collator import DataCollatorForNI


@dataclass
class GPT2Arguments(DataTrainingArguments):
    data_dir: str = field(
        default="data/splits/default/", metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    task_dir: str = field(
        default="data/tasks/", metadata={"help": "The directory for saving the NaturalInstructions tasks json files."}
    )
    output_dir: str = field(
        default="predictions/default/", metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    model_name: str = field(default="gpt2", metadata={"help": "model_name"})


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset["test"]
        
        self.data_collator = DataCollatorForNI(
            tokenizer,
            model=None,
            padding="longest",
            max_source_length=1024,
            max_target_length=128,
            add_task_definition=True,
            num_pos_examples=2,
            num_neg_examples=0,
            add_explanation=False,
            text_only=True
        )
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        encoded_example = self.data_collator([self.dataset[i]])
        return encoded_example["inputs"][0].strip()


if __name__ == "__main__":
    random.seed(123)
    parser = HfArgumentParser((GPT2Arguments,))
    args, = parser.parse_args_into_dataclasses()
    raw_datasets = load_dataset(
        "src/ni_dataset.py",
        data_dir=args.data_dir, 
        task_dir=args.task_dir, 
        max_num_instances_per_task=args.max_num_instances_per_task,
        max_num_instances_per_eval_task=args.max_num_instances_per_eval_task
    )
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    
    @torch.no_grad()
    def text_pipeline(dataset, batch_size: int = 1):
        for idx in range(0, len(dataset)):
            examples = dataset[idx]
            inputs = tokenizer(
                examples, truncation=True, padding=True, return_tensors="pt",
                max_length=1024,
            )
            output_ids = model.generate(
                **{k: v.to("cuda") for k, v in inputs.items()},
                max_new_tokens=128,
                pad_token_id=model.config.eos_token_id,
                num_beams=4,
            )
            texts = tokenizer.batch_decode(output_ids[:, inputs["input_ids"].size(1):])
            yield texts[0].strip()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "predicted_examples.jsonl"), "w") as fout:
        pbar = tqdm.tqdm(
            zip(
                raw_datasets["test"],
                text_pipeline(
                    TextDataset(raw_datasets, tokenizer),
                    batch_size=1,
                )
            ),
            total=len(raw_datasets["test"])
        )
        for example, generated in pbar:
            example["prediction"] = generated
            fout.write(json.dumps(example) + "\n")
