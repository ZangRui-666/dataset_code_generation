from huggingface_hub import login

login(token="")

# ds = load_dataset("codeparrot/github-code", streaming=True, split="train", languages=["Python"])
# print(next(iter(ds))["code"])
# from datasets import load_dataset


# For pretraining
"""
Get the data from BigQuery Using the query
SELECT
  f.repo_name, f.path, c.copies, c.size, c.content, l.license
FROM
  `bigquery-public-data.github_repos.files` AS f
JOIN
  `bigquery-public-data.github_repos.contents` AS c
ON
  f.id = c.id
JOIN
  `bigquery-public-data.github_repos.licenses` AS l
ON
  f.repo_name = l.repo_name
WHERE
  NOT c.binary
    AND ((f.path LIKE '%.py')
      AND (c.size BETWEEN 1024 AND 1048575))
"""
from datasets import load_dataset
# 15M rows
#
ds = load_dataset("transformersbook/codeparrot", streaming=True, split="train")
print(ds.info)
print(next(iter(ds))["content"])
# ds = load_dataset("codeparrot/codeparrot-clean", streaming=True, split="train")
# iter_dataset = iter(ds)
# print(next(iter_dataset)["content"])

# For fine-tuning
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("sahil2801/CodeAlpaca-20k")
def format_example(example):
    if example.get('input'):
        return f"Below is an instruction that describes a task, paired with an input that provides further context.\n\n### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        return f"Below is an instruction that describes a task.\n\n### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

# Apply the formatting
dataset = dataset.map(lambda x: {'text': format_example(x)}, remove_columns=['instruction', 'input', 'output'])
print(dataset['train'][0])
