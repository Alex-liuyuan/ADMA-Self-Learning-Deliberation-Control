"""Safe calculator tool — AST-based, replaces eval().

Supports basic arithmetic, math functions, and unit conversions
commonly needed in medical/scientific debate contexts.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from debate_rl_v2.tools.registry import ToolRegistry, ToolSchema

# Safe operator mapping
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Safe function mapping
_SAFE_FUNCS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "pi": math.pi,
    "e": math.e,
}


class _SafeEvaluator(ast.NodeVisitor):
    """AST-based safe expression evaluator."""

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in _SAFE_FUNCS:
            val = _SAFE_FUNCS[node.id]
            if isinstance(val, (int, float)):
                return val
        raise ValueError(f"Unknown variable: '{node.id}'")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return op_func(left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(self.visit(node.operand))

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are supported")
        func_name = node.func.id
        if func_name not in _SAFE_FUNCS:
            raise ValueError(f"Unknown function: '{func_name}'")
        func = _SAFE_FUNCS[func_name]
        if not callable(func):
            raise ValueError(f"'{func_name}' is not callable")
        args = [self.visit(arg) for arg in node.args]
        return func(*args)

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def safe_calculate(expression: str = "") -> str:
    """安全计算数学表达式（AST解析，无eval）。

    支持: 加减乘除、幂运算、sqrt/log/exp/ceil/floor、pi/e常量。
    """
    if not expression or not expression.strip():
        return "错误: 空表达式"
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _SafeEvaluator().visit(tree)
        # Format result
        if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
            return str(int(result))
        if isinstance(result, float):
            return f"{result:.6g}"
        return str(result)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
        return f"计算错误: {e}"
    except SyntaxError:
        return f"语法错误: 无法解析表达式 '{expression}'"


# Auto-register on import
def _register() -> None:
    registry = ToolRegistry()
    registry.register(
        name="calculator",
        description="安全计算数学表达式，支持加减乘除、幂运算、sqrt/log/exp等数学函数",
        handler=safe_calculate,
        parameters=[
            ToolSchema(
                name="expression",
                type="string",
                description="数学表达式，如 '100*0.15+50' 或 'sqrt(144)+log(100)'",
                required=True,
            ),
        ],
        category="computation",
    )


_register()
