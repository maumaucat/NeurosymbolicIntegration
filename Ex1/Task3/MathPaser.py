from binarytree import Node

operators = {
    '+': 1,
    '-': 1,
    '*': 2
}

# Function to create tokens from the given expression (make it easier to parse)
def create_tokens(expression: str) -> list:
    tokens = []
    # remove all white spaces
    expression = expression.replace(' ', '')
    expression = expression.rstrip()
    # replace all negative numbers with a special token
    i = 0
    while i < len(expression):
        if expression[i] == '-' and (i == 0 or expression[i-1] in operators or expression[i-1] == '('):
            tokens.append(expression[i] + expression[i+1])
            i += 2
        else:
            tokens.append(expression[i])
            i += 1
    return tokens

# Function to parse the expression and create a binary tree
def parse_expression(expression: str) -> Node:
    tokens = create_tokens(expression)
    # step one: calculate the reverse polish notation (RPN)
    operator_stack = []
    rpn = []
    for token in tokens:
        if token in operators:
            while operator_stack and operator_stack[-1] != '(' and operators[operator_stack[-1]] >= operators[token]:
                rpn.append(operator_stack.pop())
            operator_stack.append(token)
        elif token == '(':
            operator_stack.append(token)
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                rpn.append(operator_stack.pop())
            operator_stack.pop()
        else:
            rpn.append(token)
    while operator_stack:
        rpn.append(operator_stack.pop())

    # step two: create the binary tree from the RPN

    digit_stack = []
    for token in rpn:
        if token in operators:
            right = digit_stack.pop()
            left = digit_stack.pop()
            node = Node(token, right, left)
            digit_stack.append(node)
        else:
            node = Node(token)
            digit_stack.append(node)
    return digit_stack.pop()


# Function to evaluate the binary tree
def evaluate_tree(node: Node) -> int:
    if node.value in operators:
        left_value = evaluate_tree(node.left)
        right_value = evaluate_tree(node.right)
        if node.value == '+':
            return left_value + right_value
        elif node.value == '-':
            return  right_value - left_value
        elif node.value == '*':
            return left_value * right_value
    else:
        return int(node.value)

# Example usage / test
with open("/Datasets/math/test.txt", "r", encoding="utf-8") as file:
    for expression in file.readlines():
        print("="*50)
        print(f"Expression: {expression}")
        root = parse_expression(expression)
        print("Tree: ", root)
        print("Result: ", evaluate_tree(root))
        print("="*50)
