import ast

if "__main__" == __name__:
    with open("chunker.py", "r") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            print(f"Function name: {node.name}")
    print("Done parsing chunker.py")