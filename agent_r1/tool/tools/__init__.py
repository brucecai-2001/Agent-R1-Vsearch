"""
Specific tool implementations
"""

from agent_r1.tool.tools.search_tool import SearchTool
from agent_r1.tool.tools.calculator_tool import CalculatorTool
from agent_r1.tool.tools.wiki_search_tool import WikiSearchTool
from agent_r1.tool.tools.python_tool import PythonTool
__all__ = [
    'SearchTool',
    'CalculatorTool',
    'WikiSearchTool',
    'PythonTool',
] 

def _default_tools(env):
    if env == 'search':
        return [SearchTool()]
    elif env == 'calculator':
        return [CalculatorTool()]
    elif env == 'wikisearch':
        return [WikiSearchTool()]
    elif env == 'python':
        return [PythonTool()]
    else:
        raise NotImplementedError
