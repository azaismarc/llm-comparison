import luigi
import pandas as pd
import json
import os

from task_form_semeval import TaskFormSemEval


def format_user_prompt(prompt: str, document: str) -> str:
    """Format the prompt message with topics and document."""
    msg = prompt.format(
        document=document,
    )
    return msg

        # Loop over each email to generate a batch file
def write_payloads(df, out_path, prompt_func, system_prompt, prefix_prompt):
    """Write JSONL payloads for a given DataFrame and topics, always including system prompt."""
    with open(out_path, encoding="utf-8", mode="w") as f:
        for idx, row in df.iterrows():
            user_prompt = prompt_func(row["Sentences"])
            payload = {
                "custom_id": str(idx),
                "body": {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": prefix_prompt, "prefix": True}
                    ],
                },
                "temperature": 0
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


class TaskBuildBatch(luigi.Task):
    user_prompt_file = luigi.Parameter(default="data/prompt/semeval/user.txt")
    system_prompt_file = luigi.Parameter(default="data/prompt/semeval/system.txt")
    prefix_prompt_file = luigi.Parameter(default="data/prompt/prefix.txt")

    def requires(self):
        return TaskFormSemEval()

    def run(self):

      

        # Read prompt template
        with open(self.user_prompt_file, "r", encoding="utf-8") as f:
            user_prompt_template = f.read()

        with open(self.system_prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read()
        
        with open(self.prefix_prompt_file, "r", encoding="utf-8") as f:
            prefix_prompt = f.read()



        # Load hotel reviews
        df = pd.read_csv(self.input().path, sep='\t', index_col="ID_Review")

        
        # Full payload
        full_path = self.output().path
        write_payloads(df, full_path,
                    lambda review: format_user_prompt(user_prompt_template, review),
                    system_prompt, prefix_prompt)

    def output(self):
        return luigi.LocalTarget("data/batch/semeval_v2.jsonl")


if __name__ == "__main__":
    luigi.build([TaskBuildBatch()], local_scheduler=True)