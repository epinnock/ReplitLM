

##IMPORTS
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCasualLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from accelerate import Accelerator
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
)



#VARIABLES
LEARNING_RATE = 2e-5
EPOCHS = 3
BATCH_SIZE = 16
SEED = 42
GRADIENT_ACCUMULATION_STEPS = 1
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
MODEL_NAME = 'replit/replit-code-v1-3b'
DATASET = "sahil2801/CodeAlpaca-20k"
MIXED_PRECISION = "fp16"
USE_CPU = False
NUMBER_OF_WARMUP_STEPS = 100
MAX_LENGTH = 512
EOS_TOKEN = "</s>"
SAVE_DIRECTORY = "/"
PROMPT_DICT ={
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    
}


#FUNCTIONS

#GET DATA 
def get_data(dataset_name):
    dataset = load_dataset(dataset_name)
    return dataset

#GET TOKENIZER
def get_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    return tokenizer

#GET MODEL
def get_model(model_name,rank,lora_alpha,lora_dropout):
    model = AutoModelForCasualLM.from_pretrained(model_name, trust_remote_code=True,load_in_8bit=True, device_map="auto")
    config = LoraConfig(
    r=rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    task_type="CAUSAL_LM")
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, config)
    return model

#generate prompt
def generate_prompt(data_point,prompt_dict):
    if data_point["input"]:
        return prompt_dict["prompt_input"].format(
            instruction=data_point["instruction"], input=data_point["input"]
        )
    else:
        return prompt_dict["prompt_no_input"].format(
            instruction=data_point["instruction"]
        )
    


#generate target
def generate_target(data_point,eos_token):
    # return expected output with eos token appended
    return f"{data_point['output']}{eos_token}"

#PREPARE DATA
def prepare_data(dataset,tokenizer,batched=True):
   # Split the dataset into a train and test dataset
   prepped = dataset.map(lambda x: preprocess(x,tokenizer), batched)
   return prepped

def preprocess(data,tokenizer,prompt_dict=PROMPT_DICT,padding=True,truncation=True,max_length=512):
    # Tokenize the data
    tokenized_data = tokenizer(
        [generate_prompt(data_point,prompt_dict) for data_point in data],
        padding,
        truncation,
        max_length,
    )
    # Generate the target
    eos_token = tokenizer.eos_token
    with tokenizer.as_target_tokenizer():   
        tokenized_data["labels"] = tokenizer(
            [generate_target(data_point, eos_token) for data_point in data],
            padding,
            truncation,
            max_length,
        )["input_ids"]
    return tokenized_data

# optimizer
#TODO: Implement 8bit optimizer
def get_optimizer(model,lr=LEARNING_RATE):
    return AdamW(model.parameters(), lr)



def collate_fn(examples,tokenizer,mixed_precision=MIXED_PRECISION):
    # When using mixed precision we want round multiples of 8/16
    if mixed_precision== "fp8":
        pad_to_multiple_of = 16
    elif mixed_precision != "no":
        pad_to_multiple_of = 8
    else:
        pad_to_multiple_of = None

    return tokenizer.pad(
        examples,
        padding="longest",
        max_length=None,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors="pt",
    )


#DATA LOADER
def get_data_loader(dataset, batch_size,tokenizer, shuffle=True,collate_fn=collate_fn):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: collate_fn(x,tokenizer)
    )

def get_learning_rate_scheduler(optimizer, num_warmup_steps=0, num_training_steps=0):
    return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

#TODO : Implement saving model to hub and local
def save_model(model,tokenizer,save_dir=SAVE_DIRECTORY):
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

def training(model_name=MODEL_NAME,rank=LORA_R,alpha=LORA_ALPHA,dropout=LORA_DROPOUT,seed=SEED,dataset=DATASET,use_cpu=USE_CPU,bactch_size=BATCH_SIZE,mixed_precision=MIXED_PRECISION,epochs=EPOCHS,gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS):
    accelerator = Accelerator(cpu=use_cpu,mixed_precision=mixed_precision)
    set_seed(seed)
    dataset = get_data(dataset)
    tokenizer = get_tokenizer(model_name)
    prepped_dataset = prepare_data(dataset,tokenizer)
    dataloader= get_data_loader(prepped_dataset,bactch_size,tokenizer)
    model = get_model(model_name,rank,alpha,dropout)
    optimizer = get_optimizer(model)
    lr_scheduler = get_learning_rate_scheduler(optimizer, num_warmup_steps=0, num_training_steps=0)
    lr_scheduler, model, optimizer, dataloader = accelerator.prepare(lr_scheduler, model, optimizer, dataloader)

    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(dataloader):
            batch.to(accelerator.device)
            outputs = model(**batch)
            loss = outputs.loss
            loss /= gradient_accumulation_steps
            accelerator.backward(loss)
            if( step) % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            accelerator.print(f"Epoch {epoch}, global step {step}: loss {loss.item()}")

if __name__ == "__main__":
    training()
