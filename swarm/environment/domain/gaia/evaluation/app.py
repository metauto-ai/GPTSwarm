import os
import json
import datetime
from email.utils import parseaddr

import gradio as gr
import pandas as pd
import numpy as np

from datasets import load_dataset
from apscheduler.schedulers.background import BackgroundScheduler
from huggingface_hub import HfApi

# InfoStrings
from scorer import question_scorer
from content import format_error, format_warning, format_log, TITLE, INTRODUCTION_TEXT, CITATION_BUTTON_LABEL, CITATION_BUTTON_TEXT, model_hyperlink

TOKEN = os.environ.get("TOKEN", None)

OWNER="gaia-benchmark"
DATA_DATASET = f"{OWNER}/GAIA"
INTERNAL_DATA_DATASET = f"{OWNER}/GAIA_internal"
SUBMISSION_DATASET = f"{OWNER}/submissions_internal"
RESULTS_DATASET = f"{OWNER}/results_public"
LEADERBOARD_PATH = f"{OWNER}/leaderboard"
api = HfApi()

YEAR_VERSION = "2023"

os.makedirs("scored", exist_ok=True)

# Display the results
eval_results = load_dataset(RESULTS_DATASET, YEAR_VERSION, token=TOKEN, download_mode="force_redownload", ignore_verifications=True)
def get_dataframe_from_results(eval_results, split):
    local_df = eval_results[split]
    local_df = local_df.map(lambda row: {"model": model_hyperlink(row["url"], row["model"])})
    local_df = local_df.remove_columns(["mail", "system_prompt", "url"])
    local_df = local_df.rename_column("model", "Model name")
    local_df = local_df.rename_column("model_family", "Model family")
    local_df = local_df.rename_column("score", "Average score (%)")
    for i in [1, 2, 3]:
        local_df = local_df.rename_column(f"score_level{i}", f"Level {i} score (%)")
    df = pd.DataFrame(local_df)
    df = df.sort_values(by=["Average score (%)"], ascending=False)

    numeric_cols = [c for c in local_df.column_names if "score" in c]
    df[numeric_cols] = df[numeric_cols].multiply(100).round(decimals=2)
    #df = df.style.format("{:.2%}", subset=numeric_cols)

    return df

eval_dataframe_val = get_dataframe_from_results(eval_results=eval_results, split="validation")
eval_dataframe_test = get_dataframe_from_results(eval_results=eval_results, split="test")

# Gold answers
gold_results = {}
gold_dataset = load_dataset(INTERNAL_DATA_DATASET, f"{YEAR_VERSION}_all", token=TOKEN)
gold_results = {split: {row["task_id"]: row for row in gold_dataset[split]} for split in ["test", "validation"]}


def restart_space():
    api.restart_space(repo_id=LEADERBOARD_PATH, token=TOKEN)

TYPES = ["markdown", "number", "number", "number", "number", "str", "str"]

def add_new_eval(
    val_or_test: str,
    model: str,
    model_family: str,
    system_prompt: str,
    url: str,
    path_to_file: str,
    organisation: str,
    mail: str,
):
    # Very basic email parsing
    _, parsed_mail = parseaddr(mail)
    if not "@" in parsed_mail:
        return format_warning("Please provide a valid email adress.")

    print("Adding new eval")

    # Check if the combination model/org already exists and prints a warning message if yes
    if model.lower() in set(eval_results[val_or_test]["model"]) and organisation.lower() in set(eval_results[val_or_test]["organisation"]):
        return format_warning("This model has been already submitted.")
    
    if path_to_file is None:
        return format_warning("Please attach a file.")

    # Save submitted file
    api.upload_file(
        repo_id=SUBMISSION_DATASET, 
        path_or_fileobj=path_to_file.name, 
        path_in_repo=f"{organisation}/{model}/{YEAR_VERSION}_{val_or_test}_raw_{datetime.datetime.today()}.jsonl",
        repo_type="dataset", 
        token=TOKEN
    )

    # Compute score
    file_path = path_to_file.name        
    scores = {"all": 0, 1: 0, 2: 0, 3: 0}
    num_questions = {"all": 0, 1: 0, 2: 0, 3: 0}
    with open(f"scored/{organisation}_{model}.jsonl", "w") as scored_file:
        with open(file_path, 'r') as f:
            for ix, line in enumerate(f):
                try:
                    task = json.loads(line)
                except Exception:
                    return format_error(f"Line {ix} is incorrectly formatted. Please fix it and resubmit your file.")

                if "model_answer" not in task:
                    raise format_error(f"Line {ix} contains no model_answer key. Please fix it and resubmit your file.")
                answer = task["model_answer"]
                task_id = task["task_id"]
                try:
                    level = int(gold_results[val_or_test][task_id]["Level"])
                except KeyError:
                    return format_error(f"{task_id} not found in split {val_or_test}. Are you sure you submitted the correct file?")

                score = question_scorer(task['model_answer'], gold_results[val_or_test][task_id]["Final answer"])
                
                scored_file.write(
                    json.dumps({
                        "id": task_id,
                        "model_answer": answer,
                        "score": score,
                        "level": level
                    }) + "\n"
                )

                scores["all"] += score
                scores[level] += score
                num_questions["all"] += 1
                num_questions[level] += 1
    
    # Save scored file
    api.upload_file(
        repo_id=SUBMISSION_DATASET, 
        path_or_fileobj=f"scored/{organisation}_{model}.jsonl",
        path_in_repo=f"{organisation}/{model}/{YEAR_VERSION}_{val_or_test}_scored_{datetime.datetime.today()}.jsonl", 
        repo_type="dataset", 
        token=TOKEN
    )

    # Actual submission
    eval_entry = {
        "model": model,
        "model_family": model_family,
        "system_prompt": system_prompt,
        "url": url,
        "organisation": organisation,
        "mail": mail,
        "score": scores["all"]/num_questions["all"],
        "score_level1": scores[1]/num_questions[1],
        "score_level2": scores[2]/num_questions[2],
        "score_level3": scores[3]/num_questions[3],
    }
    eval_results[val_or_test] = eval_results[val_or_test].add_item(eval_entry)
    print(eval_results)
    eval_results.push_to_hub(RESULTS_DATASET, config_name = YEAR_VERSION, token=TOKEN)

    return format_log(f"Model {model} submitted by {organisation} successfully. \nPlease refresh the leaderboard, and wait a bit to see the score displayed")


def refresh():
    eval_results = load_dataset(RESULTS_DATASET, YEAR_VERSION, token=TOKEN, download_mode="force_redownload", ignore_verifications=True)
    eval_dataframe_val = get_dataframe_from_results(eval_results=eval_results, split="validation")
    eval_dataframe_test = get_dataframe_from_results(eval_results=eval_results, split="test")
    return eval_dataframe_val, eval_dataframe_test

def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths


demo = gr.Blocks()
with demo:
    gr.HTML(TITLE)
    gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")

    with gr.Row():
        with gr.Accordion("📙 Citation", open=False):
            citation_button = gr.Textbox(
                value=CITATION_BUTTON_TEXT,
                label=CITATION_BUTTON_LABEL,
                elem_id="citation-button",
            ) #.style(show_copy_button=True)

    with gr.Tab("Results: Validation"):
        leaderboard_table_val = gr.components.Dataframe(
            value=eval_dataframe_val, datatype=TYPES, interactive=False,
            column_widths=["20%"] 
        )
    with gr.Tab("Results: Test"):
        leaderboard_table_test = gr.components.Dataframe(
            value=eval_dataframe_test, datatype=TYPES, interactive=False,
            column_widths=["20%"] 
        )

    refresh_button = gr.Button("Refresh")
    refresh_button.click(
        refresh,
        inputs=[],
        outputs=[
            leaderboard_table_val,
            leaderboard_table_test,
        ],
    )
    with gr.Accordion("Submit a new model for evaluation"):
        with gr.Row():
            with gr.Column():
                level_of_test = gr.Radio(["validation", "test"], value="validation", label="Split")
                model_name_textbox = gr.Textbox(label="Model name")
                model_family_textbox = gr.Textbox(label="Model family")
                system_prompt_textbox = gr.Textbox(label="System prompt example")
                url_textbox = gr.Textbox(label="Url to model information")
            with gr.Column():
                organisation = gr.Textbox(label="Organisation")
                mail = gr.Textbox(label="Contact email")
                file_output = gr.File()


        submit_button = gr.Button("Submit Eval")
        submission_result = gr.Markdown()
        submit_button.click(
            add_new_eval,
            [
                level_of_test,
                model_name_textbox,
                model_family_textbox,
                system_prompt_textbox,
                url_textbox,
                file_output,
                organisation,
                mail
            ],
            submission_result,
        )

scheduler = BackgroundScheduler()
scheduler.add_job(restart_space, "interval", seconds=3600)
scheduler.start()
demo.launch(debug=True)
