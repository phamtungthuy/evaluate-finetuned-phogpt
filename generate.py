import torch
from transformers import (
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    AutoTokenizer,
)
from peft import PeftModel
from datasets import load_dataset
import time

import os
import json
filename = "./json/data.json"
split="test[1000:2000]"
if os.path.exists(filename):
    with open(filename, "r") as file:
        count = sum(1 for line in file)
else:
    count = 0
print(count)

def create_peft_config(model):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_int8_training
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,
        lora_alpha=64,
        lora_dropout=0.1,
    )

    # prepare int-8 model for training
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config

def gen(model_base, model_peft, dataset_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtyp=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_base)

    model = AutoModelForCausalLM.from_pretrained(
        model_base,
        device_map="auto",
        trust_remote_code=True,
        load_in_8bit=True,
        quantization_config=bnb_config, 
    )
    
    model.config.use_cache = False
    
    
    model = PeftModel.from_pretrained(model, model_peft)

    model.eval()
    
    test_dataset = load_dataset(dataset_id, split=split)
    PROMPT_TEMPLATE = "### Câu hỏi:\n{instruction}\n\n### Trả lời:" 
    for idx, row in enumerate(test_dataset):
        print("Current question...", idx + 1)
        if idx + 1 <= count:
            continue
        start=time.time()
        input_prompt = PROMPT_TEMPLATE.format_map(
            {"instruction": row['question']}
        )
        full_res = ""
        inputs = tokenizer(input_prompt, return_tensors="pt").to("cuda")
        for _ in range(len(input_prompt)):
            input_len = inputs["input_ids"].shape[1]
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                output = model.generate(  
                    inputs=inputs["input_ids"].to("cuda"),  
                    attention_mask=inputs["attention_mask"].to("cuda"),  
                    do_sample=True,
                    temperature=1.0,  
                    top_k=50,  
                    top_p=0.9,  
                    max_new_tokens=1,  
                    eos_token_id=tokenizer.eos_token_id,  
                    pad_token_id=tokenizer.pad_token_id  
                )
            if output[0][-1] == tokenizer.eos_token_id:
                break
            response = tokenizer.batch_decode(output[0][input_len:], skip_special_tokens=True)[:1]  
        #     response = response.split("### Trả lời:")[1]
            full_res += ''.join(response)
            inputs = {
                "input_ids": output,
                "attention_mask": torch.ones(1, len(output[0]))
            }
        obj = test_dataset[idx]
        del obj["question"]
        del obj["field"]
        obj["actual_answer"] = full_res
        with open(filename, "a") as file:
            if idx > 0:
                file.write("\n")
            file.write(json.dumps(obj, ensure_ascii=False))
        print("total generate answer: ", time.time() - start)


if __name__ == "__main__":
    gen('vinai/PhoGPT-7B5-Instruct', 'phamtungthuy/trained_law_model_2', "phamtungthuy/cauhoiphapluat_400tokenanswer")