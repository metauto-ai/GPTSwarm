#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import astunparse
from typing import List

from swarm.environment.tools.coding.executor_utils import function_with_timeout
from swarm.environment.tools.coding.executor_types import ExecuteResult, Executor
from swarm.utils.log import logger


def get_call_str(assert_statement: str) -> str:
    ast_parsed = ast.parse(assert_statement)
    try:
        call_str = ast_parsed.body[0].test.left # type: ignore
    except:
        call_str = ast_parsed.body[0].test # type: ignore

    return astunparse.unparse(call_str).strip()

def get_output(func: str, assert_statement: str, timeout: int = 5) -> str:
    try:
        exec(f"from typing import *\n{func}", globals())
        func_call = get_call_str(assert_statement)
        output = function_with_timeout(eval, (func_call, globals()), timeout)
        return output
    except TimeoutError:
        return "TIMEOUT"
    except Exception as e:
        return str(e)
    

class PyExecutor(Executor):
    def execute(self, func: str, tests: List[str], timeout: int = 5, verbose: bool = True) -> ExecuteResult:
        # Combine function code and assert statement
        imports = 'from typing import *'
        func_test_list = [f'{imports}\n{func}\n{test}' for test in tests]

        # Run the tests and collect the results
        success_tests = []
        failed_tests = []
        is_passing = True
        num_tests = len(func_test_list)
        for i in range(num_tests):

            try:
                function_with_timeout(exec, (func_test_list[i], globals()), timeout)
                success_tests.append(tests[i])
            except Exception:
                output = get_output(func, tests[i], timeout=timeout)
                failed_tests.append(f"{tests[i]} # output: {output}")
                is_passing = False

        state = [test in success_tests for test in tests]

        feedback = "Tests passed:\n" + "\n".join(success_tests) + "\n\nTests failed:"
        feedback += "\n" + "\n".join(failed_tests)
        return is_passing, feedback, tuple(state)

    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        """
        Evaluates the implementation on Human-Eval Python.

        probably should be written in a dataset-agnostic way but not now
        """
        
        code = f"""{func}

{test}

check({name})
    """
        try:
            function_with_timeout(exec, (code, globals()), timeout)
            return True
        except Exception:
            return False
        