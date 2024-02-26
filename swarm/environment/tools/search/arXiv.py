#!/usr/bin/env python
# -*- coding: utf-8 -*-

import arxiv


class ArxivSearch:
    def __init__(self):
        self.name = "ArXiv Searcher"
        self.description = "Search for a paper on ArXiv"

    def search(self, query=None, id_list=None, sort_by=arxiv.SortCriterion.Relevance, sort_order=arxiv.SortOrder.Descending):
        search = arxiv.Search(query=query, id_list=id_list, max_results=1, sort_by=sort_by, sort_order=sort_order)
        results = arxiv.Client().results(search)
        paper = next(results, None)
        
        return paper
