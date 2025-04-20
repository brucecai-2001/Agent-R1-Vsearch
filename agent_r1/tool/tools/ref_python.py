import subprocess
import tempfile
import os
import resource
from typing import Tuple, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from ...tool.tool_base import Tool

class PythonExecTool(Tool):
    """
    A Tool that executes Python code with support for scientific computing libraries.
    The tool runs in a restricted environment with limited system access.
    """

    # List of allowed modules for scientific computing
    ALLOWED_MODULES = [
        'numpy', 'pandas', 'scipy',
        'sklearn', 'math', 'random', 'statistics', 'json',
        'datetime', 'time', 're', 'collections', 'itertools'
    ]

    # Memory limit in bytes (1GB)
    MEMORY_LIMIT = 1024 * 1024 * 1024

    def __init__(self, max_workers: int = 4):
        """
        Initialize the Python execution tool
        
        Args:
            max_workers: Maximum number of parallel executions
        """
        name = "execute_python_code"
        description = "Executes Python code with support for scientific computing libraries (numpy, pandas, scipy, sklearn). The environment has limited system access for security. \
            IMPORTANT: For multiline code, all lines must be joined with '\\n' in a single line string. DO NOT use actual line breaks in the JSON."
        parameters = {
            "type": "object",
            "properties": {
                "python_code": {
                    "type": "string",
                    "description": "The Python code to execute. Use print() to output results. Available modules: numpy, pandas, matplotlib, seaborn, scipy, sklearn, and basic Python modules."
                }
            },
            "required": ["python_code"]
        }
        
        super().__init__(name, description, parameters)
        self.max_workers = max_workers
        self.timeout = 1.0  # 1 second timeout

    def execute(self, args: Dict) -> str:
        """
        Execute the Python code in a restricted environment and return the output.
        
        Args:
            args: Tool parameters, containing:
                - "python_code": Python code to execute
            
        Returns:
            Execution results as a string. Returns error if output is empty or execution times out.
        """
        python_code = args["python_code"]
        python_code_stripped = python_code.strip('"""')

        try:
            output, errors = self._run_code_in_restricted_env(python_code_stripped)
        except subprocess.TimeoutExpired:
            return "[Error]\nCode execution timed out (maximum 1 second)."
        except MemoryError:
            return "[Error]\nMemory limit exceeded (maximum 1GB)."
        
        # Check for errors
        if errors:
            return f"[Error]\n{errors}"
            
        # Check for empty output
        if not output.strip():
            return "[Error]\nNo output generated. Please use print() to output results."
            
        return output

    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        """
        Execute multiple Python codes in parallel.
        
        Args:
            args_list: List of tool parameters, each containing:
                - "python_code": Python code to execute
            
        Returns:
            List of execution results. Returns error for any empty output or timeout.
        """
        results = [""] * len(args_list)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._run_code_in_restricted_env, args["python_code"].strip('"""')): i 
                for i, args in enumerate(args_list)
            }
            
            # Process completed tasks
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    output, errors = future.result()
                    if errors:
                        results[index] = f"[Error]\n{errors}"
                    elif not output.strip():
                        results[index] = "[Error]\nNo output generated. Please use print() to output results."
                    else:
                        results[index] = output
                except subprocess.TimeoutExpired:
                    results[index] = "[Error]\nCode execution timed out (maximum 1 second)."
                except MemoryError:
                    results[index] = "[Error]\nMemory limit exceeded (maximum 1GB)."
                except Exception as e:
                    results[index] = f"[Error]\n{str(e)}"
        
        return results

    def _set_memory_limit(self):
        """Set memory limit for the current process"""
        resource.setrlimit(resource.RLIMIT_AS, (self.MEMORY_LIMIT, self.MEMORY_LIMIT))

    def _run_code_in_restricted_env(self, code: str) -> Tuple[str, str]:
        """
        Helper function that runs Python code in a restricted environment.
        """
        # Create a temporary file with the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Add security wrapper code
            wrapper_code = f"""
import sys
import os
import builtins
import resource

# Set memory limit to 1GB
resource.setrlimit(resource.RLIMIT_AS, ({self.MEMORY_LIMIT}, {self.MEMORY_LIMIT}))

# Restrict module access
ALLOWED_MODULES = {PythonExecTool.ALLOWED_MODULES}
original_import = __import__

def restricted_import(name, *args, **kwargs):
    if name in ALLOWED_MODULES:
        return original_import(name, *args, **kwargs)
    raise ImportError(f"Module {{name}} is not allowed in this environment")

# Override built-in import
builtins.__import__ = restricted_import

# Restrict system access
os.system = lambda *args, **kwargs: None
os.popen = lambda *args, **kwargs: None
os.spawn = lambda *args, **kwargs: None
os.exec = lambda *args, **kwargs: None
os.fork = lambda *args, **kwargs: None
os.kill = lambda *args, **kwargs: None

# Restrict file operations
original_open = open
def restricted_open(*args, **kwargs):
    if 'w' in kwargs.get('mode', '') or 'a' in kwargs.get('mode', ''):
        raise PermissionError("Write operations are not allowed in this environment")
    return original_open(*args, **kwargs)
builtins.open = restricted_open

try:
    # Execute user code
    {code}
except MemoryError:
    print("[Error] Memory limit exceeded", file=sys.stderr)
    sys.exit(1)
"""
            f.write(wrapper_code)
            temp_file = f.name

        process = None
        try:
            # Run the code with Python
            cmd = ["python3", temp_file]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=self._set_memory_limit  # Set memory limit for the subprocess
            )
            
            out, err = process.communicate(timeout=self.timeout)
            return out, err
        except subprocess.TimeoutExpired:
            if process:
                process.kill()  # Ensure process is terminated
                process.communicate()  # Clean up pipes
            raise
        except Exception as e:
            if process:
                process.kill()
                process.communicate()
            raise
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file)
            except:
                pass
            # Ensure process is terminated
            if process:
                try:
                    process.kill()
                except:
                    pass