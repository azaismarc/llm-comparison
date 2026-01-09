import luigi
import json

# Sentences to short or not informative enough
with open("data/id_to_remove.json", "r") as f:
    id_to_remove = set(json.load(f))

def extract_text(text: str | dict):

    if isinstance(text, dict):
        text = text['content']
    texts = text.split("\n")
    data = []
    for t in texts: # ignore header
        if "|" not in t: continue
        t = t.split("|")
        if len(t) != 3: continue
        label, reason, citation = t
        data.append({"label": label.strip(), "reason": reason.strip(), "citation": citation.strip()})
    return data

class TaskOutputFormat(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.output_path)

    def run(self):
        
        data = []
        with open(self.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        all_output_data = []
        for d in data:
            if d["id"] in id_to_remove: continue
            output_data = {
                "id": d["id"],
                "data": extract_text(d["text"])
            }
            all_output_data.append(output_data)

        with self.output().open('w') as f:
            for item in all_output_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')  # Convert dict to JSON string and write line



if __name__ == "__main__":
    from glob import glob
    tasks = []
    for input_file in glob("data/output/*/*.jsonl"):
        output_file = input_file.split("/")
        output_file[1] = "output_format"
        output_file = "/".join(output_file)
        tasks.append(TaskOutputFormat(input_path=input_file, output_path=output_file))
    luigi.build(tasks, local_scheduler=True)
