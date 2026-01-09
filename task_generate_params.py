# pipeline_generate_tasks.py
import luigi
import glob
import os
import csv

class TaskGenerateParams(luigi.Task):

    MODELS = luigi.ListParameter(default=[
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Ministral-8B-Instruct-2410",
        "mistralai/Mistral-Nemo-Instruct-FP8-2407",
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it"
    ])

    N_RUNTIME = luigi.IntParameter(default=1)
    OUTPUT_FILE = luigi.Parameter(default="batchs_topics.tsv")

    def output(self):
        return luigi.LocalTarget(self.OUTPUT_FILE)

    def run(self):
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.OUTPUT_FILE) or ".", exist_ok=True)

        # Get all input files
        input_files = glob.glob("data/batch/topics.jsonl")

        with self.output().open("w") as f:
            writer = csv.writer(f, delimiter="\t")
            # Write header
            writer.writerow(["MODEL", "INPUT_PATH", "OUTPUT_FILE"])

            # Generate rows
            for model in self.MODELS:
                for file_path in input_files:
                    basename = os.path.basename(file_path)
                    name_without_ext = os.path.splitext(basename)[0]
                    dirname = os.path.basename(os.path.dirname(file_path))

                    for i in range(1, self.N_RUNTIME + 1):
                        out_file = f"{dirname}_{name_without_ext}_v{i}.jsonl"
                        writer.writerow([model, file_path, out_file])


if __name__ == "__main__":
    # Run only this single TaskGenerateParams
    luigi.build(
        [TaskGenerateParams()],
        local_scheduler=True,
        workers=1
    )
