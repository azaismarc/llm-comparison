import luigi
import json
from glob import glob
import os
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
import pandas as pd
from itertools import combinations
from tqdm import tqdm


TOPICS = ['Chambre', 'Emplacement', 'Rapport qualité-prix', 'Ambiance', 'Personnel']

class TaskComputeIntraModel(luigi.Task):

    def output(self):
        os.makedirs("data/score", exist_ok=True)
        return luigi.LocalTarget("data/score/mattcoef_prompt.tsv")
        

    def run(self):

        input_data = dict()

        for file in glob("data/output_format/*/*.jsonl"):           
            if "batch" in file: continue
            model = file.split("/")[2]
            els = file.split("/")[3].split("_")
            v = els[0]
            annotator = '_'.join(els[1:-1])
            if "@" not in annotator: continue

            # Initialize nested dictionaries if needed
            input_data.setdefault(model, dict())
            input_data[model].setdefault(annotator, dict())
            input_data[model][annotator].setdefault(v, {t: [] for t in TOPICS})

            # Load data
            with open(file, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]

            # Fill input_data
            for d in data:
                labels = set(el.get('label') for el in d.get('data', []) if el.get('label'))
                for t in TOPICS:
                    input_data[model][annotator][v][t].append(1 if t in labels else 0)

        rows = []
        for model in tqdm(input_data):
            annotators = list(input_data[model].keys())
            for an1, an2 in list(combinations(annotators, 2)):
                for v in ['sample', 'full']:
                    row = {'Model': model, 'annotator_1': an1, 'annotator_2': an2, 'V':v}
                    rows.append(row)
                    for t in TOPICS:
                        score = matthews_corrcoef(
                            input_data[model][an1][v][t],
                            input_data[model][an2][v][t]
                        )
                        # score = cohen_kappa_score(
                        #     input_data[model][an1][v][t],
                        #     input_data[model][an2][v][t]
                        # )
                        row[t] = score
        
        df = pd.DataFrame(rows)
        df.to_csv(self.output().path, sep='\t', index=False)

if __name__ == "__main__":
   luigi.build([TaskComputeIntraModel()], local_scheduler=True)
