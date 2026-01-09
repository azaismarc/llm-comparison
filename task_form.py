import luigi
import pandas as pd
import json
import re

class TaskForm(luigi.Task):
    input_path = luigi.Parameter(default="data/form.tsv")
    output_path = luigi.Parameter(default="data/topics.json")

    def output(self):
        return luigi.LocalTarget(self.output_path)

    def run(self):
        # --- Load data ---
        
        df = pd.read_csv(self.input_path, sep="\t")

        # --- Assuming you already have a list of 'mails' from the DataFrame ---
        mails = df["Adresse e-mail"].tolist()
        data = {m: {} for m in mails}

        # --- Define helper regex and parser ---
        pattern = r'([A-ZÉÈÀ][\w\s-]*?)\s*:\s*(.*?)(?=(?:[A-ZÉÈÀ][\w\s-]*?\s*:)|$)'


        def get_topics(text: str):
            matches = re.findall(pattern, text, flags=re.MULTILINE)
            d = {}
            for topic, desc in matches:
                topic = topic.strip().title().strip()
                desc = desc.strip().lower().strip()
                d[topic] = desc
            return d

        # --- Extract standard themes ---
        themes = [
            "Chambre",
            "Emplacement",
            "Ambiance",
            "Rapport qualité-prix",
            "Personnel"
        ]

        for theme in themes:
            colname = f"Définir le thème '{theme}' :"
            if colname not in df.columns:
                continue
            values = df[colname].tolist()
            for m, v in zip(mails, values):
                data[m][theme] = v.lower()

        # --- Extract custom topics ---
        custom_col = "Ajoutez tous les thèmes que vous jugez pertinents pour comparer ses deux hôtels et définissez-les précisément :"
        if custom_col in df.columns:
            values = df[custom_col].tolist()
            for m, v in zip(mails, values):
                if isinstance(v, str):
                    d = get_topics(v)
                    for k, desc in d.items():
                        data[m][k] = desc

        # --- Save result as JSON ---
        with self.output().open("w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    luigi.build([TaskForm()], local_scheduler=True)
