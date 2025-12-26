import ast

LEETCODE_CONTEXT = """
import random
import functools
import collections
import string
import math
import datetime

from typing import *
from functools import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from string import *
from operator import *
from math import *

inf = float('inf')


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def list_node(values: list):
    if not values:
        return None
    head = ListNode(values[0])
    p = head
    for val in values[1:]:
        node = ListNode(val)
        p.next = node
        p = node
    return head


def is_same_list(p1, p2):
    if p1 is None and p2 is None:
        return True
    if not p1 or not p2:
        return False
    return p1.val == p2.val and is_same_list(p1.next, p2.next)


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def tree_node(values: list):
    if not values:
        return None
    root = TreeNode(values[0])
    i = 1
    queue = deque()
    queue.append(root)
    while queue:
        node = queue.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
            i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
            i += 1
    return root


def is_same_tree(p, q):
    if not p and not q:
        return True
    elif not p or not q:
        return False
    elif p.val != q.val:
        return False
    else:
        return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)
"""

def parse_starter_code(starter_code: str):
    """
    Extracts function name, arg names, and type hints from the solution stub.
    """
    try:
        tree = ast.parse(starter_code)
        class_def = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Solution'][0]
        func_def = [n for n in class_def.body if isinstance(n, ast.FunctionDef)][0]
        
        args = []
        for arg in func_def.args.args:
            if arg.arg == 'self': continue
            type_hint = ast.unparse(arg.annotation) if arg.annotation else "Any"
            args.append({"name": arg.arg, "type": type_hint})
            
        return_type = ast.unparse(func_def.returns) if func_def.returns else "Any"
        
        return {
            "function_name": func_def.name,
            "args": args,
            "return_type": return_type
        }
    except Exception as e:
        print(f"AST Parse Error: {e}")
        return None