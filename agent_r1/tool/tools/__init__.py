"""
Specific tool implementations
"""

from agent_r1.tool.tools.search_tool import SearchTool
from agent_r1.tool.tools.calculator_tool import CalculatorTool
from agent_r1.tool.tools.wiki_search_tool import WikiSearchTool
# from agent_r1.tool.tools.crag_api_search_tool import CragApiSearchTool
# from agent_r1.tool.tools.crag_web_search_tool import CragWebSearchTool
__all__ = [
    'SearchTool',
    'CalculatorTool',
    'WikiSearchTool',
    'CragApiSearchTool',
    'CragWebSearchTool'
] 

def _default_tools(env):
    if env == 'search':
        return [SearchTool()]
    elif env == 'calculator':
        return [CalculatorTool()]
    elif env == 'wikisearch':
        return [WikiSearchTool()]
    # elif env == 'crag_api_search':
    #     return [CragApiSearchTool()]
    # elif env == 'crag_web_search':
    #     return [CragWebSearchTool()]
    else:
        raise NotImplementedError