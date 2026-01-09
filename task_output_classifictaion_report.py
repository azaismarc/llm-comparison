import luigi
import json
import csv
from sklearn.metrics import classification_report
import pandas as pd

TOPICS = ["EMPLACEMENT","SERVICE","NOURRITURE","AMBIANCE","Rapport qualité-prix".upper()]

class TaskOutputClassificationReport(luigi.Task):
    test_path = luigi.Parameter(default="data/semeval/semeval.tsv")
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.output_path)
    
    def run(self):
        # ---- 1. Load test (gold) file and drop 'Sentences' ----
        gold = pd.read_csv(self.test_path, sep="\t", encoding="utf-8")
        gold = gold.set_index("ID_Review")[TOPICS]  # ignore Sentences

        # ---- 2. Load predictions ----
        preds = pd.read_csv(self.input_path, sep="\t", encoding="utf-8")
        preds = preds.set_index("ID_Review")[TOPICS]

        # ---- 3. Align gold and predictions ----
        gold, preds = gold.align(preds, join="inner", axis=0)

        # ---- 4. Compute classification report ----
        report_dict = classification_report(
                gold.values,
                preds.values,
                target_names=TOPICS,
                output_dict=True,
                zero_division=0
            )

        # ---- 5. Save report as JSON ----
        with self.output().open('w') as f:
            json.dump(report_dict, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    from glob import glob
    tasks = []
    for input_file in glob("data/output_semeval/*/batch_semeval_v2.tsv"):
        output_file = input_file.split("/")
        output_file = "/".join(output_file).replace("batch_semeval_v2.tsv","classification_report_v2.json")
        tasks.append(TaskOutputClassificationReport(input_path=input_file, output_path=output_file))
    luigi.build(tasks, local_scheduler=True)
