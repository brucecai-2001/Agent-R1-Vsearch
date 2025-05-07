"""
Specific tool implementations
"""

# from agent_r1.tool.tools.search_tool import SearchTool
# from agent_r1.tool.tools.calculator_tool import CalculatorTool
# from agent_r1.tool.tools.wiki_search_tool import WikiSearchTool
from agent_r1.tool.tools.vsearch import VisualSearchTool

__all__ = [
    # 'SearchTool',
    # 'CalculatorTool',
    # 'WikiSearchTool',
    'VisualSearchTool'
] 

def _default_tools(env):
    if env == 'visualsearch':
        return [VisualSearchTool()]
    else:
        return []
