#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path

def find_project_root():
    """
    Search upwards from the current working directory to find the project root.

    The project root is identified by the presence of either a '.git' folder or a
    'requirements.txt' file. This function traverses up the directory tree until
    it finds one of these markers or reaches the root directory.

    Returns:
        pathlib.Path: The path object representing the project root directory.
    """
    current_directory = Path.cwd()
    while current_directory != current_directory.parent:
        if (current_directory / ".git").exists() or (current_directory / "requirements.txt").exists():
            return current_directory
        current_directory = current_directory.parent

    raise FileNotFoundError("Project root not found.")

GPTSWARM_ROOT = find_project_root()


