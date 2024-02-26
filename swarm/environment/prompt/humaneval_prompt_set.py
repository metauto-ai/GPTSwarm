#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Any

from swarm.environment.prompt.prompt_set import PromptSet
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.environment.prompt.common import get_combine_materials


@PromptSetRegistry.register('humaneval')
class HumanEvalPromptSet(PromptSet):

    @staticmethod
    def get_role():
        return "You are an AI that only responds with only python code. "

    @staticmethod
    def get_constraint():
        return (
"You will be given a function signature and its docstring by the user. "
"Write your full implementation (restate the function signature). "
"Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
)
    @staticmethod
    def get_format():
        return "natural language"


    @staticmethod
    def get_answer_prompt(question):
        # Format the question for the AI assistant to answer
        return f"{question}"

    @staticmethod
    def get_react_prompt(question, solution, feedback):
        return f"""Here is an unsuccessful attempt for solving the folloing question:
Question:
{question}
Attempted Solution:
{solution}
Feedback:\n{feedback}
Rewrite the code based on the feedback and the following question:
{question}"""


    @staticmethod
    def get_query_prompt(question):
        return (
"# Information Gathering for Question Resolution\n\n"
"Evaluate if additional information is needed to answer the question. "
#"If web search or file analysis is required, formulate specific queries to assist in finding the answer.\n\n"
"If a web search or file analysis is necessary, outline specific clues or details to be searched for.\n\n"
f"## ❓ Target Question:\n{question}\n\n"
# "## 🤔 Information Gathering:\n"
# "Identify if a web search or file reading is necessary and outline the approach."
"## 🔍 Clues for Investigation:\n"
"Identify critical clues and concepts within the question that are essential for finding the answer.\n"
        )


    @staticmethod
    def get_file_analysis_prompt(query, file):
        return (
            # "# File Analysis Required\n\n"
            # f"## 🔍 Required Information to Extract:\n---\n{query}\n---\n\n"
            # f"## 📄 File Content for Analysis:\n---\n{file}\n---\n\n"
            # "## 🤔 Instructions:\n"
            # "Extract the specified information from the file. Example: 'Identify the main theme in the text.'"
"# File Analysis Task\n\n"
f"## 🔍 Information Extraction Objective:\n---\n{query}\n---\n\n"
f"## 📄 File Under Analysis:\n---\n{file}\n---\n\n"
"## 📝 Instructions:\n"
"1. Identify the key sections in the file relevant to the query.\n"
"2. Extract and summarize the necessary information from these sections.\n"
"3. Ensure the response is focused and directly addresses the query.\n"
"Example: 'Identify the main theme in the text.'"
        )


    @staticmethod
    def get_websearch_prompt(question, query):
        return (
            "# Web Search Task\n\n"
            f"## Original Question: \n---\n{question}\n---\n\n"
            f"## 🔍 Targeted Search Objective:\n---\n{query}\n---\n\n"
            "## 🌐 Simplified Search Instructions:\n"
            "Generate three specific search queries directly related to the original question. Each query should focus on key terms from the question. Format the output as a comma-separated list.\n"
            "For example, if the question is 'Who will be the next US president?', your queries could be: 'US presidential candidates, current US president, next US president'.\n"
            "Remember to format the queries as 'query1, query2, query3'."
        )



    @staticmethod
    def get_adversarial_answer_prompt(question):
        pass


    @staticmethod
    def get_distill_websearch_prompt(question, query, results):
        return (
            # "# Summarization of Search Results\n\n"
            # "## 🔍 Required Information for Summary:\n---\n{query}\n---\n\n"
            # "## 🌐 Search Results for Analysis:\n---\n{results}\n---\n\n"
            # "## ✏️ Instructions:\n"
            # "Summarize the key findings from the search results related to the query. "
            # "Focus on relevant information. Example: 'Summary of key points...'"
"# Summarization of Search Results\n\n"
f"## Original question: \n---\n{question}\n---\n\n"
f"## 🔍 Required Information for Summary:\n---\n{query}\n---\n\n"
f"## 🌐 Analyzed Search Results:\n---\n{results}\n---\n\n"
"## 📝 Instructions for Summarization:\n"
"1. Review the provided search results and identify the most relevant information related to the question and query.\n"
"2. Extract and highlight the key findings, facts, or data points from these results.\n"
"3. Organize the summarized information in a coherent and logical manner.\n"
"4. Ensure the summary is concise and directly addresses the query, avoiding extraneous details.\n"  
"5. If the information from web search is useless, directly answer: \"No useful information from WebSearch\".\n"  
        )


    @staticmethod
    def get_reflect_prompt(question, answer):
        return (
"# Reflection on the Task\n\n"
f"## 🤔 Reflection Question:\n---\n{question}\n---\n\n"
f"## 💡 Your Previous Answer:\n---\n{answer}\n---\n\n"
"## ✏️ Instructions:\n"
"Reflect on your answer process, considering the accuracy, method, and reasoning."
        )


    @staticmethod
    def get_self_consistency(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Answer {index + 1}: {answer}" for index, answer in enumerate(answers)])
        return (
            # "# Self-Consistency Evaluation Task\n\n"
            # f"## 🤔 Given Question:\n---\n{question}\n---\n\n"
            # "## 💡 Available Answers:\n---\n"
            # f"{formatted_answers}\n"
            # "---\n\n"
            # "## ✏️ Instructions:\n"
            # "Review the given answers and choose the most consistent one. "
            # "If all answers differ, select the one you find most reliable. "
            # f"Please keep following the constraints to answer the question: {constraint}."
"# Self-Consistency Evaluation Task\n\n"
f"## 🤔 Question for Review:\n---\n{question}\n---\n\n"
f"## 💡 Reviewable Answers:\n---\n{formatted_answers}\n---\n\n"
"## 📋 Instructions for Selection:\n"
"1. Read each answer and assess how it addresses the question.\n"
"2. Compare the answers for their adherence to the given question's criteria and logical coherence.\n"
"3. Identify the answer that best aligns with the question's requirements and is the most logically consistent.\n"
"4. Ignore the candidate answers if they do not give a direct answer, for example, using 'unable to ...', 'as an AI ...'.\n"
"5. Copy the most suitable answer as it is, without modification, to maintain its original form.\n"
f"6. Adhere to the constraints: {constraint}.\n"
"Note: If no answer fully meets the criteria, choose and copy the one that is closest to the requirements."
        )

    @staticmethod
    def get_select_best(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Answer {index + 1}: {answer}" for index, answer in enumerate(answers)])
        return (
            # "# Best Answer Evaluation Task\n\n"
            # f"## 🤔 Given Question:\n---\n{question}\n---\n\n"
            # "## 💡 Available Answers:\n---\n"
            # f"{formatted_answers}\n"
            # "---\n\n"
            # "## ✏️ Instructions:\n"
            # "Review the given question and candidate answers and choose the most reasonable one. "
            # "Please copy the original answer if you decide."
            # f"Please keep following the constraints to answer the question: {constraint}."
"# Best Answer Evaluation Task\n\n"
f"## 🤔 Question:\n---\n{question}\n---\n\n"
f"## 💡 Candidate Answers for Evaluation:\n---\n{formatted_answers}\n---\n\n"
"## 📋 Evaluation Instructions:\n"
"1. Examine the question closely to understand its requirements.\n"
"2. Read each candidate answer thoroughly and assess its relevance and accuracy about the question.\n"
"3. Choose the answer that most accurately and completely addresses the question.\n"
"4. Ignore the candidate answers if they do not give a direct answer, for example, using 'unable to ...', 'as an AI ...'.\n"
"5. Copy the chosen answer exactly as it is presented, maintaining its original format.\n"
f"6. Adhere to the constraints: {constraint}.\n"
"Note: If none of the answers fully meet the question's criteria, select the one closest to fulfilling them."
        )

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)

