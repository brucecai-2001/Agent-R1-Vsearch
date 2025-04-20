"""
Python tool implementation for executing restricted Python code
"""

import sys
import re

sys.path.append("/home/yanruiran/workspace/wdy/Agent-R1")
from agent_r1.tool.tool_base import Tool
import json
import numpy as np
import pandas as pd
import math
from typing import Dict


class PythonTool(Tool):
    """
    Tool for executing restricted Python code
    """

    def __init__(self):
        """
        Initialize Python tool
        """
        name = "python"
        description = "Execute Python code with support for scientific computing. Includes numpy (as np) for numerical operations, pandas (as pd) for data analysis, and math module for mathematical functions. Provides basic Python built-ins like abs, len, max, min, etc. For safety, system operations, file operations, and new imports are not allowed."
        parameters = {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": """Python code to execute. Available features:
                    - Pre-imported modules: numpy (as np), pandas (as pd), math
                    - Built-in functions: abs, all, any, bool, dict, float, int, len, list, max, min, range, round, str, sum, tuple
                    - Multi-line code support with automatic return of last expression
                    - No imports, system operations, or file operations allowed
                    - Maximum memory usage: 1GB
                    Example: 'arr = np.array([1, 2, 3]); arr.mean()'""",
                }
            },
            "required": ["code"],
        }
        super().__init__(name, description, parameters)

    def is_safe_code(self, code: str) -> bool:
        """
        Check if the code is safe to execute
        """
        # Add limit of large array operations
        dangerous_patterns = [
            r"np\.ones\(\s*\d{6,}",  # Check large array creation
            r"np\.zeros\(\s*\d{6,}",
            r"np\.array\(\s*\[\s*(?:[^]]*,\s*){1000,}",  # Check large list
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                return False

        # List of dangerous keywords
        dangerous_keywords = [
            "exec",
            "eval",
            "open",
            "file",
            "system",
            "subprocess",
            "os",
            "sys",
            "shutil",
            "globals",
            "locals",
            "compile",
        ]

        # Check for dangerous keywords
        for keyword in dangerous_keywords:
            if keyword in code:
                return False

        return True

    def execute(self, args: Dict) -> str:
        """
        Execute Python code
        """
        code = args.get("code", "").strip()

        resource_limit = """
def limit_memory(maxsize):
    import resource
    import sys
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

limit_memory(1024 * 1024 * 1024)  # Memory limit: 1GB
"""

        if not code:
            return json.dumps({"error": "No code provided"})

        # Check code safety
        if not self.is_safe_code(code):
            return json.dumps({"error": "Code contains unsafe operations"})

        try:
            # Create namespace with pre-imported modules
            namespace = {
                "np": np,
                "pd": pd,
                "math": math,
                "abs": abs,
                "all": all,
                "any": any,
                "bool": bool,
                "dict": dict,
                "float": float,
                "int": int,
                "len": len,
                "list": list,
                "max": max,
                "min": min,
                "range": range,
                "round": round,
                "str": str,
                "sum": sum,
                "tuple": tuple,
            }

            # Add resource limit
            code = resource_limit + "\n" + code

            # Execute code in the namespace
            exec(code, namespace)

            # Get value of last expression
            lines = code.strip().split("\n")
            if lines:
                last_line = lines[-1].strip()
                try:
                    result = eval(last_line, namespace)
                    # Add result length limit
                    result_str = str(result)
                    if len(result_str) > 10000:  # Set a reasonable length limit
                        result_str = result_str[:10000] + "... (output truncated)"
                    return json.dumps({"result": result_str})
                except:
                    return json.dumps(
                        {"result": "Code executed successfully, but no return value"}
                    )

        except Exception as e:
            return json.dumps({"error": f"Execution error: {str(e)}"})

    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate reward score for Python code execution

        Args:
            args: Tool parameters
            result: Tool execution result

        Returns:
            Reward value
        """
        # Parse result
        try:
            result_dict = json.loads(result)
        except:
            return 0.0

        # If execution failed
        if "error" in result_dict:
            return 0.1  # Small reward for attempting

        # Base reward
        reward = 0.5

        # Increase reward based on code complexity
        code = args.get("code", "")
        # Calculate number of lines
        num_lines = len(code.strip().split("\n"))
        # Add reward based on lines (max 0.3)
        complexity_reward = min(0.3, 0.05 * num_lines)

        # Additional reward for having return value
        if (
            "result" in result_dict
            and result_dict["result"]
            != "Code executed successfully, but no return value"
        ):
            reward += 0.2

        return min(1.0, reward + complexity_reward)  # Cap total reward at 1.0


if __name__ == "__main__":
    python_tool = PythonTool()

    # Test case 1: Simple arithmetic
    test_code1 = """
x = 10
y = 20
x + y
"""
    result1 = python_tool.execute({"code": test_code1})
    reward1 = python_tool.calculate_reward({"code": test_code1}, result1)
    print("Test 1 - Simple arithmetic:")
    print("Result:", result1)
    print(
        "Reward:", reward1
    )  # Expected: ~0.85 (0.5 base + 0.05*3 lines + 0.2 return value)
    print()

    # Test case 2: NumPy array operations
    test_code2 = """
arr = np.array([1, 2, 3, 4, 5])
mean = arr.mean()
std = arr.std()
mean + std
"""
    result2 = python_tool.execute({"code": test_code2})
    reward2 = python_tool.calculate_reward({"code": test_code2}, result2)
    print("Test 2 - NumPy operations:")
    print("Result:", result2)
    print("Reward:", reward2)  # Expected: ~0.9 (0.5 base + 0.2 return + 0.2 complexity)
    print()

    # Test case 3: Math module usage
    test_code3 = """
radius = 5
area = math.pi * radius ** 2
area
"""
    result3 = python_tool.execute({"code": test_code3})
    reward3 = python_tool.calculate_reward({"code": test_code3}, result3)
    print("Test 3 - Math module:")
    print("Result:", result3)
    print(
        "Reward:", reward3
    )  # Expected: ~0.85 (0.5 base + 0.15 complexity + 0.2 return)
    print()

    # Test case 4: Error case (unsafe operation)
    test_code4 = """
import os
os.system('ls')
"""
    result4 = python_tool.execute({"code": test_code4})
    reward4 = python_tool.calculate_reward({"code": test_code4}, result4)
    print("Test 4 - Unsafe operation:")
    print("Result:", result4)
    print("Reward:", reward4)  # Expected: 0.1 (error case)
    print()

    # Test case 5: Complex numpy operations
    test_code5 = """
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
eigenvalues = np.linalg.eigvals(matrix)
np.sort(eigenvalues)
"""
    result5 = python_tool.execute({"code": test_code5})
    reward5 = python_tool.calculate_reward({"code": test_code5}, result5)
    print("Test 5 - Complex NumPy:")
    print("Result:", result5)
    print(
        "Reward:", reward5
    )  # Expected: ~0.85 (0.5 base + 0.15 complexity + 0.2 return)
    print()

    # Test case 6: No return value
    test_code6 = """
x = 100
y = 200
z = x + y
"""
    result6 = python_tool.execute({"code": test_code6})
    reward6 = python_tool.calculate_reward({"code": test_code6}, result6)
    print("Test 6 - No return value:")
    print("Result:", result6)
    print("Reward:", reward6)  # Expected: ~0.65 (0.5 base + 0.15 complexity)
    print()

    # Test case 7: Large output handling
    test_code7 = """
arr = np.arange(10000)
arr
"""
    result7 = python_tool.execute({"code": test_code7})
    reward7 = python_tool.calculate_reward({"code": test_code7}, result7)
    print("Test 7 - Large output:")
    print("Result:", result7)
    print("Reward:", reward7)  # Expected: ~0.85 (0.5 base + 0.2 return + 0.15 complexity)
    print()

    # Test case 8: Memory intensive operation
    test_code8 = """
matrix = np.random.rand(1000000, 1000000)
matrix.sum()
"""
    result8 = python_tool.execute({"code": test_code8})
    reward8 = python_tool.calculate_reward({"code": test_code8}, result8)
    print("Test 8 - Memory intensive:")
    print("Result:", result8)
    print("Reward:", reward8)  # Expected: ~0.85 (0.5 base + 0.2 return + 0.15 complexity)
    print()

