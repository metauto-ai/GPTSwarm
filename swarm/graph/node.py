#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shortuuid
import asyncio
from typing import List, Any, Optional
from abc import ABC, abstractmethod
import warnings

from swarm.memory import GlobalMemory
from swarm.utils.log import logger
import pdb


class Node(ABC):
    """
    Represents a processing unit within a graph-based framework.

    This class encapsulates the functionality for a node in a graph, managing
    connections to other nodes, handling inputs and outputs, and executing
    assigned operations asynchronously. It supports both individual and
    aggregated processing modes.

    Attributes:
        id (uuid.UUID): Unique identifier for the node.
        agent: Associated agent for node-specific operations.
        operation_description (str): Brief description of the node's operation.
        predecessors (List[Node]): Nodes that precede this node in the graph.
        successors (List[Node]): Nodes that succeed this node in the graph.
        inputs (List[Any]): Inputs to be processed by the node.
        outputs (List[Any]): Results produced after node execution.
        is_aggregate (bool): Indicates if node aggregates inputs before processing.

    Methods:
        add_predecessor(operation): 
            Adds a node as a predecessor of this node, establishing a directed connection.
        add_successor(operation): 
            Adds a node as a successor of this node, establishing a directed connection.
        execute(**kwargs): 
            Asynchronously processes the inputs through the node's operation, handling each input individually.
        _execute(input, **kwargs): 
            An internal method that defines how a single input is processed by the node. This method should be implemented specifically for each node type.
    """

    def __init__(self, #agent: Type, 
                 operation_description: str, 
                 id: Optional[str], combine_inputs_as_one: bool,
                 ):
        """
        Initializes a new Node instance.
        """
        self.id = id if id is not None else shortuuid.ShortUUID().random(length=4)
        self.memory = GlobalMemory.instance()
        self.operation_description = operation_description
        self.predecessors: List[Node] = []
        self.successors: List[Node] = []
        self.inputs: List[Any] = []
        self.outputs: List[Any] = []
        self.combine_inputs_as_one = combine_inputs_as_one

    @property
    def node_name(self):
        return self.__class__.__name__
    
    def add_predecessor(self, operation: 'Node'):

        if operation not in self.predecessors:
            self.predecessors.append(operation)
            operation.successors.append(self)

    def add_successor(self, operation: 'Node'):

        if operation not in self.successors:
            self.successors.append(operation)
            operation.predecessors.append(self)

    def remove_predecessor(self, operation: 'Node'):
        if operation in self.predecessors:
            self.predecessors.remove(operation)
            operation.successors.remove(self)

    def remove_successor(self, operation: 'Node'):
        if operation in self.successors:
            self.successors.remove(operation)
            operation.predecessors.remove(self)

    def process_input(self, inputs):

        all_inputs = []
        if inputs is None:
            if self.predecessors:

                for predecessor in self.predecessors:
                    predecessor_input = self.memory.query_by_id(predecessor.id)

                    if isinstance(predecessor_input, list) and predecessor_input:
                        predecessor_input = predecessor_input[-1]
                        all_inputs.append(predecessor_input)
                inputs = all_inputs
            else:
                raise ValueError("Input must be provided either directly or from predecessors.")
            
        elif not isinstance(inputs, list):

            inputs = [inputs]

        return inputs

    async def execute(self, **kwargs):

        self.outputs = []
        tasks = []
        if not self.inputs and self.predecessors:
            if self.combine_inputs_as_one:
                combined_inputs = []
                for predecessor in self.predecessors:
                    predecessor_outputs = predecessor.outputs
                    if predecessor_outputs is not None and isinstance(predecessor_outputs, list):
                        combined_inputs.extend(predecessor_outputs)
                tasks.append(asyncio.create_task(self._execute(combined_inputs, **kwargs)))
            else:
                for predecessor in self.predecessors:
                    predecessor_outputs = predecessor.outputs
                    if isinstance(predecessor_outputs, list) and predecessor_outputs:
                        for predecessor_output in predecessor_outputs:
                            tasks.append(asyncio.create_task(self._execute(predecessor_output, **kwargs)))
        elif self.inputs:
            tasks = [asyncio.create_task(self._execute(input, **kwargs)) for input in self.inputs]
        else:
            warnings.warn("No input received.")
            return

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if not isinstance(result, Exception):
                    if not isinstance(result, list):
                        result = [result]
                    self.outputs.extend(result)
                else:
                    logger.error(f"Node {type(self).__name__} failed to execute due to: {result.__class__.__name__}: {result}")

    @abstractmethod
    async def _execute(self, input, **kwargs):
        """ To be overriden by the descendant class """

    def log(self):

        items_for_id = self.memory.query_by_id(self.id)

        if items_for_id:
            last_item = items_for_id[-1]
        else:
            last_item = {}

        ignore_keys = ['task', 'input', 'format'] 
        formatted_items = '\n    '.join(
            f"\033[1;34m{key}\033[0m: {value}" for key, value in last_item.items() if key not in ignore_keys)
        formatted_output = f"Memory Records for ID \033[1;35m{self.id}\033[0m:\n    {formatted_items}"
        logger.info(formatted_output)

