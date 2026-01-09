import luigi
import json
import os
import time
import transformers
import torch
from tqdm import tqdm
from vllm import LLM
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
import torch
# Permet à PyTorch d’utiliser la totalité de la VRAM
torch.cuda.set_per_process_memory_fraction(1.0)
torch.backends.cudnn.benchmark = True  # optimisation des kernels
device = torch.device("cuda:0")

# Test rapide pour voir la VRAM disponible
print(f"VRAM totale GPU : {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")


os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import login
hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

login(token=hf_token)


class TaskRun(luigi.Task):
    MODEL = luigi.Parameter()
    input_path = luigi.Parameter()
    output_file = luigi.Parameter()

    def run(self):

        custom_ids = []
        messages_list = []

        # Open and read the JSONL file line by line
        with open(self.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                # Extract custom_id
                custom_ids.append(obj.get('custom_id'))
                # Extract messages (list of dicts)
                messages_list.append(obj['body'].get('messages', []))
        
        if "70B" not in self.MODEL:
            # Prepare LLM
            if 'mistral' in self.MODEL.lower():
                llm = LLM(
                    model=self.MODEL,
                    tokenizer_mode="mistral",
                    load_format="mistral",
                    config_format="mistral",
                    gpu_memory_utilization=0.9,  # safer
                    tensor_parallel_size=2
                )
            else:
                llm = LLM(
                    model=self.MODEL,
                    gpu_memory_utilization=0.9,  # safer
                    tensor_parallel_size=2
                )
        

            # structured_output_params = StructuredOutputsParams(regex=r"# Réponse:.*")
            # sampling_params = SamplingParams(temperature=0.0, structured_outputs=structured_output_params, max_tokens=512)

            # res = llm.chat(messages=messages_list, sampling_params=sampling_params)

            sampling_params = SamplingParams(temperature=0.0,  max_tokens=512)

            res = llm.chat(messages=messages_list, sampling_params=sampling_params)

            output_data = []
            for custom_id, r in zip(custom_ids, res):
                output_data.append({"id": custom_id, "text": r.outputs[0].text})
            del llm
        else:
            pipeline = transformers.pipeline(
                "text-generation",
                model=self.MODEL,
                device_map="auto",
                model_kwargs={
                    "torch_dtype": torch.bfloat16, 
                    
                }
            )
            pipeline.model.set_attn_implementation("flash_attention_2")

            torch.compile(pipeline.model)

            output_data = []
            for custom_id, messages in tqdm(list(zip(custom_ids, messages_list))):
                outputs = pipeline(
                    messages,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.0   
                )
                output_data.append({"id": custom_id, "text":outputs[0]["generated_text"][-1]["content"]})
            
        
        
        with self.output().open('w') as f:
            for item in output_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(5)

    def output(self):
        model_name = self.MODEL.split("/")[-1]
        return luigi.LocalTarget(
            f"data/output/{model_name}/{os.path.basename(self.output_file)}"
        )

if __name__ == "__main__":
    import sys
    import csv
    tsv_file = sys.argv[1]
    task_index = int(sys.argv[2])  # zero-based index in TSV (excluding header)

    with open(tsv_file, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f, delimiter="\t"))
        params = reader[task_index]

    luigi.build(
        [TaskRun(
            MODEL=params["MODEL"].strip(),
            input_path=params["INPUT_PATH"],
            output_file=params["OUTPUT_FILE"]
        )],
        local_scheduler=True,
        workers=1
    )