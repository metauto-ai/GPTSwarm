#!/usr/bin/env python
# -*- coding: utf-8 -*-
# modified based on https://github.com/princeton-nlp/tree-of-thought-llm/blob/ab400345c5ea39d28ea6d7d3be0e417b11113c87/scripts/crosswords/search_crosswords-dfs.ipynb
import re


def parse_response(response):
    def parse_line(input_str):
        # regular expression pattern to match the input string format
        pattern = r'^([hv][1-5])\. ([a-zA-Z]{5,5}) \((certain|high|medium|low)\).*$'

        # use regex to extract the parts of the input string
        match = re.match(pattern, input_str)

        if match:
            # extract the matched groups
            parts = [match.group(1), match.group(2), match.group(3)]
            return parts
        else:
            return None
    # split the response into lines
    lines = response.split('\n')

    # parse each line
    parsed_lines = [parse_line(line) for line in lines]

    # filter out the lines that didn't match the format
    confidence_to_value = {'certain': 1, 'high': 0.5, 'medium': 0.2, 'low': 0.1} 
    parsed_lines = [(line[0].lower() + '. ' + line[1].lower(), confidence_to_value.get(line[2], 0)) for line in parsed_lines if line is not None]
    return sorted(parsed_lines, key=lambda x: x[1], reverse=True)