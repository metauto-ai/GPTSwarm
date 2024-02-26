#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import torch
import pickle 
from collections import deque, defaultdict
from typing import Any, List, Optional, Deque, Dict
from swarm.utils.log import logger
from swarm.utils.globals import Singleton

class Memory:
    """
    A memory storage system that maintains a collection of items, each represented as a dictionary.
    Provides functionalities to add, retrieve, and query items in memory. Supports querying
    by key, content, ID, and semantic similarity (if RAG is enabled).

    Methods:
        items: Property that returns a list of all items in memory.
        add: Adds an item to the memory and indexes it by the given ID.
        get: Retrieves an item from memory by its index.
        query_by_key: Retrieves items that contain the specified key.
        query_by_content: Retrieves items whose contents match the given key-value pairs.
        query_by_id: Retrieves items associated with a specific identifier.
        query_by_similarity: Retrieves items semantically similar to a given query, based on a similarity threshold.
        clear: Clears all items and indices from the memory.

    Attributes:
        use_rag (bool): Flag to use Retrieval-Augmented Generation (RAG) for semantic similarity queries.

    Args:
        use_rag (bool): Flag to enable or disable RAG for semantic similarity queries.
    """

    def __init__(self, use_rag: bool = False) -> None:
        self._items: Dict[str, List[Dict[str, Any]]] = {}
        self.use_rag = use_rag
        if use_rag:
            from transformers import BertTokenizer, BertModel
            from scipy.spatial.distance import cosine
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased')

    @property
    def items(self) -> List[Dict[str, Any]]:
        return list(self._items)

    def add(self, id: str, item: Dict[str, Any]) -> None:
        if id not in self._items:
            self._items[id] = []
        self._items[id].append(item)

    def get(self, index: int) -> Dict[str, Any]:
        return self._items[index]

    def query_by_key(self, key: str) -> List[Dict[str, Any]]:

        return [item for item in self._items if key in item]

    def query_by_operations(self, operation: str) -> List[Dict[str, Any]]:
        return [item for id, items in self._items.items() for item in items if item.get('operation') == operation]

    def query_by_content(self, **kwargs) -> List[Dict[str, Any]]:
        return [
            item for item in self._items 
            if all(str(value).lower() in str(item.get(key, '')).lower() for key, value in kwargs.items())]

    def query_by_id(self, id: str) -> List[Dict[str, Any]]:
        return self._items.get(id, [])
    
    def query_by_similarity(self, query: str, threshold: float = 0.5) -> List[Dict[str, Any]]:

        if not self.use_rag:
            raise RuntimeError("Semantic similarity query requires 'use_rag' to be True. Set 'use_rag=True' to use this feature.")

        logger.info("Calculating and retrieving most similar information...")
        from scipy.spatial.distance import cosine

        inputs = self.tokenizer(query, return_tensors='pt')
        outputs = self.model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

        results = []
        for item in self._items:
            for key, value in item.items():
                if isinstance(value, str):
                    item_inputs = self.tokenizer(value, return_tensors='pt')
                    item_outputs = self.model(**item_inputs)
                    item_embedding = item_outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

                    similarity = 1 - cosine(query_embedding, item_embedding)
                    print(similarity)
                    if similarity > threshold:
                        results.append(item)
                        break
        return results


    def clear(self) -> None:
        self._items.clear()

    def __repr__(self) -> str:

        def format_item(item):
            return '\n    '.join(f"\033[1;34m{key}\033[0m: {value}" for key, value in item.items())
        def format_items_for_id(id, items):
            return f"\033[1;35m{id}\033[0m:\n    " + '\n    '.join(format_item(item) for item in items)

        class_name = f"\033[1;32m{self.__class__.__name__}\033[0m" 
        contents = "\033[1;31mContents:\033[0m" 
        formatted_items = '\n  '.join(format_items_for_id(id, items) for id, items in self._items.items())

        return f"{class_name} {contents}\n  " + formatted_items

class GlobalMemory(Memory, Singleton):
    def __init__(self, use_rag: bool = False):
        Memory.__init__(self, use_rag) 
        Singleton.__init__(self)
