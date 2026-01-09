import luigi
import pandas as pd
import json
import os

from task_form import TaskForm


def format_user_prompt(prompt: str, document: str, topics: dict) -> str:
    """Format the prompt message with topics and document."""
    topics_formatted = "\n".join(f"{t} : {desc}" for t, desc in topics.items())
    msg = prompt.format(
        document=document,
        topics=topics_formatted
    )
    return msg

        # Loop over each email to generate a batch file
def write_payloads(df, out_path, topics, prompt_func, system_prompt, prefix_prompt):
    """Write JSONL payloads for a given DataFrame and topics, always including system prompt."""
    with open(out_path, encoding="utf-8", mode="w") as f:
        for idx, row in df.iterrows():
            user_prompt = prompt_func(row.review, topics)
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

set_topics = ['Chambre', 'Emplacement', 'Rapport qualité-prix', 'Ambiance', 'Personnel']

class TaskBuildBatch(luigi.Task):
    user_prompt_file = luigi.Parameter(default="data/prompt/user.txt")
    system_prompt_file = luigi.Parameter(default="data/prompt/system.txt")
    prefix_prompt_file = luigi.Parameter(default="data/prompt/prefix.txt")
    reviews_file = luigi.Parameter(default="data/hotels.tsv")

    def requires(self):
        return TaskForm()

    def run(self):

        for k, v in self.output().items():
            os.makedirs(v.path, exist_ok=True)

        # Read prompt template
        with open(self.user_prompt_file, "r", encoding="utf-8") as f:
            user_prompt_template = f.read()

        with open(self.system_prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read()
        
        with open(self.prefix_prompt_file, "r", encoding="utf-8") as f:
            prefix_prompt = f.read()

        # Load topics per email
        with self.input().open() as f:
            topics_by_email = json.load(f)

        # Load hotel reviews
        df = pd.read_csv(self.reviews_file, sep='\t', index_col="id")

        for email, topics in topics_by_email.items():
            # Full payload
            full_path = os.path.join(self.output()['full'].path, f"{email}.jsonl")
            write_payloads(df, full_path, topics,
                        lambda review, t: format_user_prompt(user_prompt_template, review, t),
                        system_prompt, prefix_prompt)

            # Sample payload with filtered topics
            topics_filtered = {k: v for k, v in topics.items() if k in set_topics}
            sample_path = os.path.join(self.output()['sample'].path, f"{email}.jsonl")
            write_payloads(df, sample_path, topics_filtered,
                        lambda review, t: format_user_prompt(user_prompt_template, review, t),
                        system_prompt, prefix_prompt)

    def output(self):
        return {
            'full': luigi.LocalTarget("data/batch/full"),
            'sample': luigi.LocalTarget("data/batch/sample")
        }

if __name__ == "__main__":
    luigi.build([TaskBuildBatch()], local_scheduler=True)