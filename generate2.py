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
split="test[2000:3000]"
if os.path.exists(filename):
    with open(filename, "r") as file:
        count = sum(1 for line in file)
else:
    count = 0
print(count)

def gen(model_base, model_peft, dataset_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtyp=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_base, trust_remote_code=True)

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
        input_ids = tokenizer(input_prompt, return_tensors="pt")
        outputs = model.generate(
            inputs=input_ids["input_ids"].to("cuda"),
            attention_mask=input_ids["attention_mask"].to("cuda"),
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=0.9,
            max_new_tokens=400,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]  
        response = response.split("### Trả lời:")[1]
        obj = test_dataset[idx]
        del obj["question"]
        del obj["field"]
        obj["actual_answer"] = response
        with open(filename, "a") as file:
            if idx > 0:
                file.write("\n")
            file.write(json.dumps(obj, ensure_ascii=False))
            count += 1
        print("total generation time ", time.time() - start)


if __name__ == "__main__":
    gen('vinai/PhoGPT-7B5-Instruct', 'phamtungthuy/law-model-version2', "phamtungthuy/cauhoiphapluat_400tokenanswer")