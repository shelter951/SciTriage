from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Iterable, List


def audit_torchvision_test_label_leak(source: str, dataset_name: str = "CIFAR10") -> Dict[str, object]:
    """Detect loops that read labels from a torchvision test split."""
    tree = ast.parse(source)
    visitor = _TorchvisionTestLabelLeakVisitor(dataset_name)
    visitor.visit(tree)
    return {
        "passed": len(visitor.label_leaks) == 0,
        "test_label_leaks": visitor.label_leaks,
        "test_dataset_vars": sorted(visitor.test_dataset_vars),
    }


def audit_hf_test_label_leak(
    source: str,
    dataset_name: str,
    test_split: str = "test",
    label_key: str = "label",
) -> Dict[str, object]:
    """Detect loops that read label fields from a HuggingFace test split."""
    tree = ast.parse(source)
    visitor = _HfTestLabelLeakVisitor(dataset_name, test_split, label_key)
    visitor.visit(tree)
    return {
        "passed": len(visitor.leaks) == 0,
        "test_label_leaks": visitor.leaks,
        "dataset_vars": sorted(visitor.dataset_vars),
    }


def audit_checkpoint_artifacts(path: str | Path, required: Iterable[str] | None = None) -> Dict[str, object]:
    """Check whether a checkpoint directory contains required artifacts."""
    root = Path(path)
    names = list(required or ["best.pkl", "spec_list.pkl", "model_params.pkl"])
    present = {name: (root / name).exists() for name in names}
    return {
        "passed": all(present.values()),
        "present": present,
        "root": str(root),
    }


class _TorchvisionTestLabelLeakVisitor(ast.NodeVisitor):
    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        self.test_dataset_vars: set[str] = set()
        self.label_leaks: List[Dict[str, object]] = []

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        if _is_torchvision_test_call(node.value, self.dataset_name):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.test_dataset_vars.add(target.id)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        iter_name = _iter_dataset_name(node.iter)
        if iter_name in self.test_dataset_vars:
            label_names = _label_names_from_tuple_target(node.target)
            used = _load_names(node.body)
            leaked = sorted(label_names & used)
            if leaked:
                self.label_leaks.append({
                    "line": node.lineno,
                    "dataset": iter_name,
                    "label_names": leaked,
                })
        self.generic_visit(node)


class _HfTestLabelLeakVisitor(ast.NodeVisitor):
    def __init__(self, dataset_name: str, test_split: str, label_key: str) -> None:
        self.dataset_name = dataset_name
        self.test_split = test_split
        self.label_key = label_key
        self.dataset_vars: set[str] = set()
        self.leaks: List[Dict[str, object]] = []

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        if _is_hf_load_dataset_call(node.value, self.dataset_name):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.dataset_vars.add(target.id)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        split = _iter_dataset_split(node.iter)
        if split and split["dataset"] in self.dataset_vars and split["split"] == self.test_split:
            row_names = _row_names_from_target(node.target)
            if _body_reads_label_key(node.body, row_names, self.label_key):
                self.leaks.append({
                    "line": node.lineno,
                    "dataset": split["dataset"],
                    "split": split["split"],
                    "row_names": sorted(row_names),
                    "label_key": self.label_key,
                })
        self.generic_visit(node)


class _LoadNameFinder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)


def _is_torchvision_test_call(node: ast.AST, dataset_name: str) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    is_dataset = (
        isinstance(func, ast.Name) and func.id == dataset_name
    ) or (
        isinstance(func, ast.Attribute) and func.attr == dataset_name
    )
    if not is_dataset:
        return False
    for keyword in node.keywords:
        if keyword.arg == "train" and isinstance(keyword.value, ast.Constant):
            return keyword.value.value is False
    return False


def _is_hf_load_dataset_call(node: ast.AST, dataset_name: str) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    is_load_dataset = (
        isinstance(func, ast.Name) and func.id == "load_dataset"
    ) or (
        isinstance(func, ast.Attribute) and func.attr == "load_dataset"
    )
    if not is_load_dataset or not node.args:
        return False
    return isinstance(node.args[0], ast.Constant) and node.args[0].value == dataset_name


def _iter_dataset_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "enumerate"
        and node.args
        and isinstance(node.args[0], ast.Name)
    ):
        return node.args[0].id
    return None


def _iter_dataset_split(node: ast.AST) -> Dict[str, str] | None:
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and _slice_value(node.slice) is not None
    ):
        return {"dataset": node.value.id, "split": str(_slice_value(node.slice))}
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "enumerate"
        and node.args
    ):
        return _iter_dataset_split(node.args[0])
    return None


def _slice_value(node: ast.AST) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    return None


def _label_names_from_tuple_target(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Tuple) and len(target.elts) >= 2 and isinstance(target.elts[1], ast.Tuple):
        return _label_names_from_tuple_target(target.elts[1])
    if isinstance(target, ast.Tuple) and len(target.elts) >= 2:
        label = target.elts[1]
        if isinstance(label, ast.Name):
            return {label.id}
        if isinstance(label, ast.Tuple):
            return {elt.id for elt in label.elts if isinstance(elt, ast.Name)}
    return set()


def _row_names_from_target(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Tuple):
        names = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
        return set(names[-1:])
    return set()


def _body_reads_label_key(nodes: Iterable[ast.AST], row_names: set[str], label_key: str) -> bool:
    for node in nodes:
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Subscript)
                and isinstance(child.value, ast.Name)
                and child.value.id in row_names
                and _slice_value(child.slice) == label_key
            ):
                return True
    return False


def _load_names(nodes: Iterable[ast.AST]) -> set[str]:
    finder = _LoadNameFinder()
    for node in nodes:
        finder.visit(node)
    return finder.names
