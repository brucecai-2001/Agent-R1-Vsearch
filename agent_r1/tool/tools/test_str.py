

def add_prefix_tab(code: str) -> str:
    """
    Add prefix tab to multi-line code
    """
    lines = code.strip().split("\n")
    if len(lines) > 1:
        return "\n".join([f"\t{line}" for line in lines])
    return code

code = add_prefix_tab("""
x = 10
y = 20
print(x + y)
""")

template = f"""
try:
{code}
except Exception as e:
    print(e)
"""

tmp_file = "tmp.py"

with open(tmp_file, "w") as f:
    f.write(template)

# print(template.format(code=code))
