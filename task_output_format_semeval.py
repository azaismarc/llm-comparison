import luigi
import json
import csv

TOPICS = ["EMPLACEMENT","SERVICE","NOURRITURE","AMBIANCE","Rapport qualité-prix".upper()]

class TaskOutputFormatSemEval(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.output_path)

    def run(self):
        # Read input JSONL
        data = []
        with open(self.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))

        # Prepare header
        HEADER = ["ID_Review"] + TOPICS
        rows = [HEADER]

        # Build rows
        for d in data:
            review_id = d.get("id", "")
            topic_labels = {t: 0 for t in TOPICS}

            for el in d.get("data", []):
                label = el.get("label")
                if label.strip().upper() in topic_labels:
                    topic_labels[label.strip().upper()] = 1  # mark topic as present

            row = [review_id] + [topic_labels[t] for t in TOPICS]
            rows.append(row)

        # Write CSV output
        with self.output().open('w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(rows)



if __name__ == "__main__":
    from glob import glob
    tasks = []
    for input_file in glob("data/output_format/*/batch_semeval_v2.jsonl"):
        output_file = input_file.split("/")
        output_file[1] = "output_semeval"
        output_file = "/".join(output_file).replace("jsonl","tsv")
        tasks.append(TaskOutputFormatSemEval(input_path=input_file, output_path=output_file))
    luigi.build(tasks, local_scheduler=True)
