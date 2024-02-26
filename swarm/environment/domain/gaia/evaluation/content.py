TITLE = """<h1 align="center" id="space-title">GAIA Leaderboard</h1>"""

INTRODUCTION_TEXT = """
GAIA is a benchmark which aims at evaluating next-generation LLMs (LLMs with augmented capabilities due to added tooling, efficient prompting, access to search, etc). (See our [paper](https://arxiv.org/abs/2311.12983) for more details.)

## Data
GAIA is made of more than 450 non-trivial question with an unambiguous answer, requiring different levels of tooling and autonomy to solve. 
It is therefore divided in 3 levels, where level 1 should be breakable by very good LLMs, and level 3 indicate a strong jump in model capabilities. Each level is divided into a fully public dev set for validation, and a test set with private answers and metadata. 

GAIA data can be found in [this dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA). Questions are contained in `metadata.jsonl`. Some questions come with an additional file, that can be found in the same folder and whose id is given in the field `file_name`.

## Submissions
Results can be submitted for both validation and test. Scores are expressed as the percentage of correct answers for a given split. 

We expect submissions to be json-line files with the following format. The first two fields are mandatory, `reasoning_trace` is optionnal:
```
{"task_id": "task_id_1", "model_answer": "Answer 1 from your model", "reasoning_trace": "The different steps by which your model reached answer 1"}
{"task_id": "task_id_2", "model_answer": "Answer 2 from your model", "reasoning_trace": "The different steps by which your model reached answer 2"}
```
Submission made by our team are labelled "GAIA authors". While we report average scores over different runs when possible in our paper, we only report the best run in the leaderboard.

**Please do not repost the public dev set, nor use it in training data for your models.**
"""

CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"
CITATION_BUTTON_TEXT = r"""@misc{mialon2023gaia,
      title={GAIA: a benchmark for General AI Assistants}, 
      author={Grégoire Mialon and Clémentine Fourrier and Craig Swift and Thomas Wolf and Yann LeCun and Thomas Scialom},
      year={2023},
      eprint={2311.12983},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}"""


def format_error(msg):
    return f"<p style='color: red; font-size: 20px; text-align: center;'>{msg}</p>"

def format_warning(msg):
    return f"<p style='color: orange; font-size: 20px; text-align: center;'>{msg}</p>"

def format_log(msg):
    return f"<p style='color: green; font-size: 20px; text-align: center;'>{msg}</p>"

def model_hyperlink(link, model_name):
    return f'<a target="_blank" href="{link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{model_name}</a>'

