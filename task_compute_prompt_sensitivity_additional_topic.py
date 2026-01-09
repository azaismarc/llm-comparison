import luigi
import json
from glob import glob
import os
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
import pandas as pd
import krippendorff
import numpy as np

TOPICS = ['Chambre', 'Emplacement', 'Rapport qualité-prix', 'Ambiance', 'Personnel']

class TaskComputePromptSensitivityAdditionalTopic(luigi.Task):

    def output(self):
        os.makedirs("data/score", exist_ok=True)
        return {
            'kappa': luigi.LocalTarget("data/score/mattcoef_annotator_per_model.tsv"),
            'kripp': luigi.LocalTarget("data/score/mattcoef_annotator_per_model.tsv")
        }

    def run(self):
        input_data = dict()
        input_kappa = dict()
        input_krip = dict()

        for file in glob("data/output_format/*/*.jsonl"): 
            if "batch" in file: continue
            model = file.split("/")[2]
            els = file.split("/")[3].split("_")
            v = els[0]
            annotator = '_'.join(els[1:-1])
            if "@" not in annotator: continue

            # Initialize nested dictionaries if needed
            input_data.setdefault(model, dict())
            input_kappa.setdefault(model, dict())
            input_krip.setdefault(model, dict())
            input_data[model].setdefault(annotator, dict())
            input_kappa[model].setdefault(annotator, dict())
            input_data[model][annotator].setdefault(v, {t: [] for t in TOPICS})
            input_krip[model].setdefault(v, {t: [] for t in TOPICS})

            # Load data
            with open(file, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]

            # Fill input_data
            for d in data:
                labels = set(el.get('label') for el in d.get('data', []) if el.get('label'))
                for t in TOPICS:
                    input_data[model][annotator][v][t].append(1 if t in labels else 0)

        for model in input_data:
            annotators = list(input_data[model].keys())
            for v in ['sample', 'full']:
                for t in TOPICS:
                    values = [input_data[model][a][v][t] for a in annotators]
                    input_krip[model][v][t] = krippendorff.alpha(values, level_of_measurement="nominal")
                  

            for annotator in annotators:
                # Ensure both 'sample' and 'full' exist
                if 'sample' in input_data[model][annotator] and 'full' in input_data[model][annotator]:
                    for t in TOPICS:
                        # score = matthews_corrcoef(
                        #     input_data[model][annotator]['sample'][t],
                        #     input_data[model][annotator]['full'][t]
                        # )
                        score = matthews_corrcoef(
                            input_data[model][annotator]['sample'][t],
                            input_data[model][annotator]['full'][t]
                        )
                        input_kappa[model][annotator][t] = score

        rows = []
        for model, annotators in input_kappa.items():
            for annotator, topics in annotators.items():
                row = {'Model': model, 'Annotator': annotator}
                row.update(topics)
                rows.append(row)
            
        

        df = pd.DataFrame(rows)
        df.to_csv(self.output()['kappa'].path, sep='\t', index=False)

        rows = []
        for model, v in input_krip.items():
            for v, topics in v.items():
                row = {'Model': model, 'V': v}
                row.update(topics)
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(self.output()['kripp'].path, sep='\t', index=False)



if __name__ == "__main__":
   luigi.build([TaskComputePromptSensitivityAdditionalTopic()], local_scheduler=True)
