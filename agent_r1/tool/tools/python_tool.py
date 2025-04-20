"""
Python tool implementation for executing restricted Python code
"""

import sys
import re
import os
import json
import tempfile
import subprocess
import resource
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

sys.path.append("/home/yanruiran/workspace/wdy/Agent-R1")
from agent_r1.tool.tool_base import Tool


class PythonTool(Tool):
    """
    Tool for executing restricted Python code
    """

    # 允许的模块列表
    ALLOWED_MODULES = [
        "numpy",
        "numpy.core",
        "numpy.core._methods",
        "numpy.lib",
        "numpy.linalg",
        "pandas",
        "math",
        "scipy",
        "sklearn",
        "random",
        "statistics",
        "json",
        "datetime",
        "time",
        "re",
        "collections",
        "itertools",
    ]

    # 内存限制（1GB）
    MEMORY_LIMIT = 1024 * 1024 * 1024

    def __init__(self, max_workers: int = 4):
        """
        Initialize Python tool
        """
        name = "python"
        description = "Executes Python code with support for scientific computing libraries (numpy, pandas, scipy, sklearn). The environment has limited system access for security. \
            IMPORTANT: For multiline code, all lines must be joined with '\\n' in a single line string. DO NOT use actual line breaks in the JSON."
        parameters = {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute. Use print() to output results. Available modules: numpy, pandas, matplotlib, seaborn, scipy, sklearn, and basic Python modules.",
                }
            },
            "required": ["code"],
        }

        super().__init__(name, description, parameters)
        self.max_workers = max_workers
        self.timeout = 5.0  # 5秒超时

    def is_safe_code(self, code: str) -> bool:
        """
        Check if the code is safe to execute
        """
        # 限制大型数组操作
        dangerous_patterns = [
            r"np\.ones\(\s*\d{6,}",  # 检查大数组创建
            r"np\.zeros\(\s*\d{6,}",
            r"np\.array\(\s*\[\s*(?:[^]]*,\s*){1000,}",  # 检查大列表
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                return False

        # 危险关键词列表
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

        # 检查危险关键词
        for keyword in dangerous_keywords:
            if keyword in code:
                return False

        return True

    def add_prefix_tab(self, code: str) -> str:
        """
        Add prefix tab to (multi-line) code
        """
        lines = code.strip().split("\n")
        return "\n".join([f"\t{line}" for line in lines])

    def execute(self, args: Dict) -> str:
        """
        Execute Python code in a restricted environment
        """
        code = args.get("code", "").strip()

        if not code:
            return json.dumps({"error": "No code provided"})

        # 检查代码安全性
        if not self.is_safe_code(code):
            return json.dumps({"error": "Code contains unsafe operations"})

        code = self.add_prefix_tab(code)

        try:
            output, errors = self._run_code_in_restricted_env(code)
        except subprocess.TimeoutExpired:
            return json.dumps(
                {"error": f"Code execution timed out (maximum {self.timeout} seconds)"}
            )
        except MemoryError:
            return json.dumps({"error": "Memory limit exceeded (maximum 1GB)"})
        except Exception as e:
            return json.dumps({"error": f"Execution error: {str(e)}"})

        # 检查错误
        if errors:
            return json.dumps({"error": f"Execution error: {errors}"})

        # 如果output为空但没有错误，表示代码执行成功但没有输出
        if not output.strip():
            return json.dumps(
                {
                    "result": "Code executed successfully, but no output. Use print() to show results."
                }
            )

        # 返回结果
        # 结果长度限制
        if len(output) > 10000:
            output = output[:10000] + "... (output truncated)"
        return json.dumps({"result": output})

    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        """
        Execute multiple Python codes in parallel
        """
        results = [""] * len(args_list)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(
                    self._run_code_in_restricted_env,
                    self.add_prefix_tab(args.get("code", "").strip()),
                ): i
                for i, args in enumerate(args_list)
            }

            # 处理完成的任务
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    output, errors = future.result()

                    # 检查错误
                    if errors:
                        results[index] = json.dumps(
                            {"error": f"Execution error: {errors}"}
                        )
                    # 如果output为空但没有错误
                    elif not output.strip():
                        results[index] = json.dumps(
                            {
                                "result": "Code executed successfully, but no output. Use print() to show results."
                            }
                        )
                    else:
                        results[index] = json.dumps({"result": output})
                except subprocess.TimeoutExpired:
                    results[index] = json.dumps(
                        {
                            "error": f"Code execution timed out (maximum {self.timeout} seconds)"
                        }
                    )
                except MemoryError:
                    results[index] = json.dumps(
                        {"error": "Memory limit exceeded (maximum 1GB)"}
                    )
                except Exception as e:
                    results[index] = json.dumps({"error": f"Execution error: {str(e)}"})

        return results

    def _set_memory_limit(self):
        """Set memory limit for the current process"""
        resource.setrlimit(resource.RLIMIT_AS, (self.MEMORY_LIMIT, self.MEMORY_LIMIT))

    def _run_code_in_restricted_env(self, code: str) -> Tuple[str, str]:
        """
        Helper function that runs Python code in a restricted environment using a subprocess
        """
        # 创建临时文件来存放代码
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # 添加安全包装代码
            wrapper_code = f"""
import sys
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
import builtins
import resource
import numpy as np
import pandas as pd
import math

# 设置内存限制为1GB
resource.setrlimit(resource.RLIMIT_AS, ({self.MEMORY_LIMIT}, {self.MEMORY_LIMIT}))

# 限制模块访问
ALLOWED_MODULES = {self.ALLOWED_MODULES}
original_import = __import__

def restricted_import(name, *args, **kwargs):
    # 检查模块名称是否在允许列表中
    if name in ALLOWED_MODULES:
        return original_import(name, *args, **kwargs)
    
    # 检查是否是已允许模块的子模块，很重要，例如 numpy 会调用 numpy.core 等
    for allowed in ALLOWED_MODULES:
        if name.startswith(allowed + "."):
            return original_import(name, *args, **kwargs)
            
    raise ImportError(f"Module {{name}} is not allowed in this environment")

# 覆盖内置import
builtins.__import__ = restricted_import

# 限制系统访问
if 'os' in globals():
    os.system = lambda *args, **kwargs: None
    os.popen = lambda *args, **kwargs: None
    os.spawn = lambda *args, **kwargs: None
    os.exec = lambda *args, **kwargs: None
    os.fork = lambda *args, **kwargs: None
    os.kill = lambda *args, **kwargs: None

# 限制文件操作
original_open = open
def restricted_open(*args, **kwargs):
    if 'w' in kwargs.get('mode', '') or 'a' in kwargs.get('mode', ''):
        raise PermissionError("Write operations are not allowed in this environment")
    return original_open(*args, **kwargs)
builtins.open = restricted_open

# 用户代码
try:
{code}
except MemoryError:
    print("[Error] Memory limit exceeded", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"[Error] {{type(e).__name__}}: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
            f.write(wrapper_code)
            # print(f"[Debug]wrapper_code: {wrapper_code}")
            temp_file = f.name

        process = None
        try:
            # 使用Python运行代码
            cmd = ["python3", temp_file]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=self._set_memory_limit,  # 为子进程设置内存限制
            )

            out, err = process.communicate(timeout=self.timeout)
            return out, err
        except subprocess.TimeoutExpired:
            if process:
                process.kill()  # 确保进程被终止
                process.communicate()  # 清理管道
            raise
        except Exception as e:
            if process:
                process.kill()
                process.communicate()
            raise
        finally:
            # 清理临时文件
            try:
                os.unlink(temp_file)
            except:
                pass
            # 确保进程被终止
            if process:
                try:
                    process.kill()
                except:
                    pass

    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate reward score for Python code execution

        Args:
            args: Tool parameters
            result: Tool execution result

        Returns:
            Reward value
        """
        # 解析结果
        try:
            result_dict = json.loads(result)
        except:
            return 0.0

        print(result_dict)

        # 如果执行失败
        if "error" in result_dict:
            return 0.1  # 小奖励(用于尝试)

        # 基础奖励
        reward = 0.5

        # 根据代码复杂度增加奖励
        code = args.get("code", "")
        # 计算行数
        num_lines = len(code.strip().split("\n"))
        # 根据行数添加奖励(最大0.3)
        complexity_reward = min(0.3, 0.05 * num_lines)

        # 有输出的额外奖励
        if (
            "result" in result_dict
            and result_dict["result"]
            != "Code executed successfully, but no output. Use print() to show results."
            and len(result_dict["result"]) > 0
        ):
            reward += 0.2

        return min(1.0, reward + complexity_reward)  # 总奖励上限为1.0


if __name__ == "__main__":
    python_tool = PythonTool()

    # 准备所有测试用例代码
    test_code1 = """
x = 10
y = 20
print(x + y)
"""
    test_code2 = """
arr = np.array([1, 2, 3, 4, 5])
mean = arr.mean()
std = arr.std()
print(f"Mean: {mean}, Std: {std}")
print(f"Sum: {mean + std}")
"""
    test_code3 = """
radius = 5
area = math.pi * radius ** 2
print(f"Area of circle with radius {radius}: {area}")
"""
    test_code4 = """
import os
os.system('ls')
"""
    test_code5 = """
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
eigenvalues = np.linalg.eigvals(matrix)
print("Eigenvalues:", eigenvalues)
print("Sorted eigenvalues:", np.sort(eigenvalues))
"""
    test_code6 = """
x = 100
y = 200
z = x + y
print(f"Result: {z}")
"""
    test_code7 = """
arr = np.arange(100)
print("Array created")
print(arr)
"""

    # 将所有测试用例打包为批处理
    all_test_codes = [
        {"code": test_code1},
        {"code": test_code2},
        {"code": test_code3},
        {"code": test_code4},
        {"code": test_code5},
        {"code": test_code6},
        {"code": test_code7},
    ]

    # 使用batch_execute一次性执行所有测试
    results = python_tool.batch_execute(all_test_codes)

    # 处理并打印测试结果
    test_names = [
        "Test 1 - Simple arithmetic:",
        "Test 2 - NumPy operations:",
        "Test 3 - Math module:",
        "Test 4 - Unsafe operation:",
        "Test 5 - Complex NumPy:",
        "Test 6 - Using print:",
        "Test 7 - Large output:",
    ]

    test_codes = [
        test_code1,
        test_code2,
        test_code3,
        test_code4,
        test_code5,
        test_code6,
        test_code7,
    ]

    for i, (result, name, code) in enumerate(zip(results, test_names, test_codes)):
        reward = python_tool.calculate_reward({"code": code}, result)
        print(name)
        print("Result:", result)
        print("Reward:", reward)
        print()
