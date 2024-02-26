#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from typing import Any, Dict, List, Tuple, Union
from sentence_transformers import SentenceTransformer, util

from swarm.llm.format import Message
from swarm.llm import LLMRegistry
from swarm.memory.memory import GlobalMemory
from swarm.environment.prompt.gaia_prompt_set import GaiaPromptSet
from swarm.utils.const import GPTSWARM_ROOT
from swarm.utils.log import logger
from swarm.environment.domain.gaia import question_scorer

class MetaPromptOptimizer:
    """
    A class for optimizing meta prompts for language models.

    Methods:
        evaluator(answer, gt): Scores an answer against a ground truth.
        generate(task, input_constraint, objective): Generates meta constraints for a task.
        process_records(records, task_embedding, sample_type): Processes records to find relevant samples.
        create_prompt(objective, initial_constraint, samples): Creates a detailed prompt for meta-instructions.
        save_meta_prompt(data, file_name): Saves meta prompt data to a file.
        read_existing_data(file_path): Reads existing meta prompt data from a file.
    """

    def __init__(self, domain: str, model_name: str, operation: str):
        self.domain = domain
        self.model_name = model_name
        self.operation = operation
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm = LLMRegistry.get(model_name)
        self.memory = GlobalMemory.instance()
        self.positive_samples = []
        self.negative_samples = []
        self.all_samples = []


    async def generate(self,  
                       init_prompt: str, 
                       init_constraint: str, 
                       init_role: str,
                       tests: dict,
                       tests_num: int = 2,
                       max_tries: int = 5,
                       data_desc: str = "python code generation",
                       objective: str = "These META-INSTRUCTIONS will be used as references to generate specific prompts for an LLM, leading to targeted outputs.") -> str:
        
        """
        # Retrive the samples
        previous_records = self.memory.query_by_operations(self.operation)

        related_samples = self.process_records(previous_records, task, "recent")

        if previous_records:
            initial_constraint = previous_records[-1].get("constraint", "")
        else:
            initial_constraint = input_constraint

        logger.info(f"\ninitial_constraint:\n {initial_constraint}")

        prompt = self.create_prompt(objective, initial_constraint, related_samples)

        logger.info(f"\nprompt:\n{prompt}")

        instruction = [Message(role="system", content="Start with 'META-INSTRUCTIONS:'"), 
                       Message(role="user", content=prompt)]

        meta_constraint = await self.llm.agen(instruction, max_tokens=200)
        meta_constraint = meta_constraint.split("META-INSTRUCTIONS:")[-1].strip() if isinstance(meta_constraint, str) else ""

        official_constraint = GaiaPromptSet.get_gaia_constraint()
        cosine_scores = util.pytorch_cos_sim(self.embedding_model.encode(meta_constraint, convert_to_tensor=True), 
                                             self.embedding_model.encode(official_constraint, convert_to_tensor=True)).item()


        MetaPromptOptimizer.save_meta_prompt({"meta_constraint": meta_constraint, 
                                              "cosine_similarity": cosine_scores}, 
                                              "meta_constraint_data70.json")
        """

        prompt = f"""
        [Original Instructions] for {data_desc}:
        {init_constraint}

        Your objective is to refine the [Original Instructions] to better solve the task.
        Please ensure the revised instructions are concise and more effective for {data_desc}.

        META-INSTRUCTIONS:
        """

        instruction = [Message(role="system", content="You are a meta-prompt designer. Your answer should start with 'META-INSTRUCTIONS:'"), 
                       Message(role="user", content=prompt)]

        try_idx = 0
        while try_idx <= max_tries:

            try_idx += 1
            meta_constraint = await self.llm.agen(instruction, max_tokens=200)  
            meta_constraint = meta_constraint.split("META-INSTRUCTIONS:")[-1].strip() if isinstance(meta_constraint, str) else ""
            is_solved, feedback = await self.meta_evaluator(init_prompt, init_role, meta_constraint, tests)  #tests[:tests_num]
            
            if is_solved:
                return meta_constraint

        return init_constraint


    def process_records(self, records: List[Dict], task, sample_type: str) -> Union[List[str], str, None]:

        pass

        """
        task_embedding = self.embedding_model.encode(task, convert_to_tensor=True)

        most_relevant_sample = None
        highest_similarity = -1

        for i, record in enumerate(records):
            try:
                # Problem solving
                is_solved = self.evaluator(record['output'], record['ground_truth'])

                # Compared to official constraint
                previous_constraint = record["constraint"]
                official_constraint = GaiaPromptSet.get_gaia_constraint()
                
                cosine_scores = util.pytorch_cos_sim(self.embedding_model.encode(previous_constraint, convert_to_tensor=True), 
                                                    self.embedding_model.encode(official_constraint, convert_to_tensor=True)).item()

                #current_detail = (f"Task {i}: \"{record['task']}\"\nUsed Constraint: \"{record['constraint']}\"\nYour Answer: \"{record['output']}\"\nGT Answer: \"{record['ground_truth']}\"\nYour Answer is {is_solved}\nThe score of the used constraint: {cosine_scores}")
                execution_details = (
                    f"Task {i}: \"{record['task']}\"\n\n"
                    f"Used Constraints: \"{record['constraint']}\"\n\n"
                    f"Your Answer: \"{record['output']}\"\n\n"
                    f"GT Answer: \"{record['ground_truth']}\"\n\n"
                    f"Your Answer is {is_solved}"
                )

                if is_solved:
                    self.positive_samples.append(execution_details)
                else:
                    self.negative_samples.append(execution_details)

                # Task Similarity
                record_embedding = self.embedding_model.encode(record['task'], convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(task_embedding, record_embedding).item()

                if similarity > highest_similarity:
                    highest_similarity = similarity
                    most_relevant_sample = execution_details

                self.all_samples.append(execution_details)

            except KeyError as e:
                logger.error(f"Missing key in record: {e}")

        most_recent_sample = self.all_samples[-3:] if self.all_samples else ["None"]

        logger.info(most_recent_sample)

        if sample_type == "similar":
            return most_relevant_sample
        elif sample_type == "recent":
            return most_recent_sample
        elif sample_type == "positive":
            return self.positive_samples
        elif sample_type == "negative":
            return self.negative_samples
        else:
            return None
        """
            

    def create_prompt(self, objective: str, initial_constraint: str, samples: list) -> str:

        samples = '\n\n'.join(samples)

        task_type = "QA"

        prompt_template = """
        Your task is to create a set of short and precise META-INSTRUCTIONS. These instructions should clearly outline the expected format and content for responses, with a focus on simplicity and adherence to specific rules.

        Study the examples provided below, which consist of previous queries and their corresponding desired responses. These examples serve as illustrations and may not cover all potential tasks. Analyze them to understand how responses should be structured for precision and clarity.

        ```
        {samples}
        ```

        Your goal is to distill the key principles from these examples and formulate a comprehensive set of META-INSTRUCTIONS that ensure subsequent LLM responses strictly adhere to the specified format and content requirements, promoting clarity and precision.

        Please avoid directly copying the examples. Instead, use them as references to create a unique set of META-INSTRUCTIONS.
        """


        return prompt_template.format(task_type=task_type, initial_constraint=initial_constraint, samples=samples)

    async def meta_evaluator(self, prompt: str, role: str, constraint: str, tests):

        message = [Message(role="system", content=f"{role}{constraint}"),
                Message(role="user", content=prompt)]
        
        answer = await self.llm.agen(message)

        from swarm.environment.tools.coding.python_executor import PyExecutor

        is_solved, feedback, _ = PyExecutor().execute(answer, [tests])

        return is_solved, feedback


    @staticmethod
    def save_meta_prompt(data: Dict[str, Any], file_name: str) -> None:
        try:
            file_path = os.path.join(GPTSWARM_ROOT, "meta_constraint", file_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            existing_data = MetaPromptOptimizer.read_existing_data(file_path)
            data_with_id = {"id": len(existing_data) + 1, **data}
            existing_data.append(data_with_id)
            with open(file_path, "w") as file:
                json.dump(existing_data, file, indent=4)
            logger.info(f"Data saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")

    @staticmethod
    def read_existing_data(file_path) -> List[Dict]:
        try:
            with open(file_path, "r") as file:
                return json.load(file)
        except (IOError, json.JSONDecodeError):
            logger.error("Failed to read or decode existing data. Creating new file.")
            return []

