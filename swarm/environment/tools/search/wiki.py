#!/usr/bin/env python
# -*- coding: utf-8 -*-

import wikipedia

class WikiSearch:
    def __init__(self):
        self.name = "Wikipedia SearchEngine"
        self.description = "Seach for an item in Wikipedia"

    def search(self, query):
        result = wikipedia.search(query[:300], results=1, suggestion=True)
        print(result)
        if len(result[0]) != 0:
            return wikipedia.page(title=result[0]).content
        
        if result[1] is not None:
            result = wikipedia.search(result[1], results=1)
            return wikipedia.page(title=result[0]).content
        
        return None
