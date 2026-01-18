"""
Validation Gates for IR and Mermaid.
Compiler-style validation with retry mechanism.
"""

from typing import List, Tuple, Optional
import re
import json


class ValidationError(Exception):
    """Base validation error."""
    pass


class IRValidationError(ValidationError):
    """IR validation error."""
    pass


class MermaidSyntaxError(ValidationError):
    """Mermaid syntax error."""
    pass


class MermaidParseError(ValidationError):
    """Mermaid parse error."""
    pass


class MermaidRenderError(ValidationError):
    """Mermaid render error."""
    pass


class LLMComplianceError(ValidationError):
    """LLM compliance error."""
    pass


class Validator:
    """Validation gates for IR and Mermaid."""
    
    def __init__(self):
        """Initialize validator."""
        pass
    
    def validate_ir(self, ir_model: dict) -> Tuple[bool, List[str]]:
        """
        Validate PseudoCodeModel IR.
        
        Args:
            ir_model: PseudoCodeModel dictionary
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # JSON schema validation
        required_fields = ["entry_function", "file", "steps", "edges"]
        for field in required_fields:
            if field not in ir_model:
                errors.append(f"Missing required field: {field}")
        
        if "steps" in ir_model and not isinstance(ir_model["steps"], list):
            errors.append("'steps' must be a list")
        
        if "edges" in ir_model and not isinstance(ir_model["edges"], list):
            errors.append("'edges' must be a list")
        
        # Graph connectivity validation
        if "steps" in ir_model and "edges" in ir_model:
            step_ids = {s["id"] for s in ir_model["steps"]}
            
            # Check all edges reference valid nodes
            for edge in ir_model["edges"]:
                if "from" not in edge or "to" not in edge:
                    errors.append("Edge missing 'from' or 'to' field")
                elif edge["from"] not in step_ids:
                    errors.append(f"Edge references unknown 'from' node: {edge['from']}")
                elif edge["to"] not in step_ids:
                    errors.append(f"Edge references unknown 'to' node: {edge['to']}")
        
        # Loop correctness validation
        if "steps" in ir_model and "edges" in ir_model:
            loop_ids = {s["id"] for s in ir_model["steps"] if s["type"] == "loop"}
            for loop_id in loop_ids:
                loop_edges = [e for e in ir_model["edges"] if e["from"] == loop_id]
                if len(loop_edges) < 2:
                    errors.append(f"Loop node {loop_id} must have at least 2 outgoing edges")
        
        # Branch completeness
        if "steps" in ir_model and "edges" in ir_model:
            decision_ids = {s["id"] for s in ir_model["steps"] if s["type"] == "decision"}
            for decision_id in decision_ids:
                decision_edges = [e for e in ir_model["edges"] if e["from"] == decision_id]
                if len(decision_edges) < 2:
                    errors.append(f"Decision node {decision_id} must have at least 2 outgoing edges")
                # Check for Yes/No labels
                labels = [e.get("label", "") for e in decision_edges]
                if "YES" not in labels and "Yes" not in labels and "TRUE" not in labels:
                    errors.append(f"Decision node {decision_id} missing YES/True branch")
                if "NO" not in labels and "No" not in labels and "FALSE" not in labels:
                    errors.append(f"Decision node {decision_id} missing NO/False branch")
        
        # Switch correctness
        if "steps" in ir_model and "edges" in ir_model:
            switch_ids = {s["id"] for s in ir_model["steps"] if s["type"] == "switch"}
            for switch_id in switch_ids:
                switch_edges = [e for e in ir_model["edges"] if e["from"] == switch_id]
                if len(switch_edges) == 0:
                    errors.append(f"Switch node {switch_id} must have at least one outgoing edge")
        
        # Return correctness
        if "steps" in ir_model and "edges" in ir_model:
            return_ids = {s["id"] for s in ir_model["steps"] if s["type"] == "return"}
            end_ids = {s["id"] for s in ir_model["steps"] if s["type"] == "end"}
            if end_ids:
                end_id = list(end_ids)[0]
                for return_id in return_ids:
                    return_edges = [e for e in ir_model["edges"] if e["from"] == return_id]
                    if not any(e["to"] == end_id for e in return_edges):
                        errors.append(f"Return node {return_id} must connect to end node")
        
        # Break/continue correctness
        if "steps" in ir_model and "edges" in ir_model:
            break_ids = {s["id"] for s in ir_model["steps"] if s["type"] == "break"}
            continue_ids = {s["id"] for s in ir_model["steps"] if s["type"] == "continue"}
            
            # Break should exit loop or switch
            for break_id in break_ids:
                break_edges = [e for e in ir_model["edges"] if e["from"] == break_id]
                if len(break_edges) == 0:
                    errors.append(f"Break node {break_id} must have outgoing edge")
            
            # Continue should jump to loop condition
            for continue_id in continue_ids:
                continue_edges = [e for e in ir_model["edges"] if e["from"] == continue_id]
                if len(continue_edges) == 0:
                    errors.append(f"Continue node {continue_id} must have outgoing edge")
        
        # Structural validation
        if "steps" in ir_model:
            start_nodes = [s for s in ir_model["steps"] if s["type"] == "start"]
            if len(start_nodes) != 1:
                errors.append(f"Expected exactly one start node, found {len(start_nodes)}")
            
            end_nodes = [s for s in ir_model["steps"] if s["type"] == "end"]
            if len(end_nodes) != 1:
                errors.append(f"Expected exactly one end node, found {len(end_nodes)}")
        
        # All nodes reachable
        if "steps" in ir_model and "edges" in ir_model:
            start_nodes = [s for s in ir_model["steps"] if s["type"] == "start"]
            if start_nodes:
                start_id = start_nodes[0]["id"]
                reachable = self._get_reachable_nodes(start_id, ir_model["edges"])
                all_ids = {s["id"] for s in ir_model["steps"]}
                unreachable = all_ids - reachable
                if unreachable:
                    errors.append(f"Unreachable nodes: {unreachable}")
        
        return len(errors) == 0, errors
    
    def _get_reachable_nodes(self, start_id: str, edges: List[dict]) -> set:
        """Get all nodes reachable from start_id."""
        reachable = set()
        queue = [start_id]
        
        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue
            reachable.add(current)
            
            for edge in edges:
                if edge["from"] == current and edge["to"] not in reachable:
                    queue.append(edge["to"])
        
        return reachable
    
    def validate_mermaid(self, mermaid_code: str) -> Tuple[bool, List[str]]:
        """
        Validate Mermaid flowchart code.
        
        Args:
            mermaid_code: Mermaid code string
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check for LLM reasoning or invalid syntax
        if not mermaid_code or not mermaid_code.strip():
            errors.append("Mermaid code is empty")
            return False, errors
        
        # Remove markdown code blocks if present
        mermaid_code = mermaid_code.strip()
        if mermaid_code.startswith("```"):
            lines = mermaid_code.split("\n")
            mermaid_code = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        
        # Check for common LLM errors
        if "```" in mermaid_code:
            errors.append("Mermaid code contains markdown code blocks")
        
        # Check for reasoning text
        reasoning_keywords = ["here", "following", "converted", "flowchart", "based on", "according to"]
        first_line = mermaid_code.split("\n")[0].lower()
        if any(keyword in first_line for keyword in reasoning_keywords) and not first_line.startswith("flowchart"):
            errors.append("Mermaid code may contain reasoning text")
        
        # Syntax validation
        if not mermaid_code.startswith("flowchart"):
            errors.append("Mermaid code must start with 'flowchart TD'")
        
        # Check for valid node declarations
        node_pattern = r'[A-Za-z0-9_]+\[.*?\]|[A-Za-z0-9_]+\(.*?\)|[A-Za-z0-9_]+\{.*?\}'
        nodes = re.findall(node_pattern, mermaid_code)
        if len(nodes) < 2:  # At least start and end
            errors.append("Mermaid code must contain at least 2 nodes")
        
        # Check for valid arrows
        arrow_pattern = r'-->|--\|'
        arrows = re.findall(arrow_pattern, mermaid_code)
        if len(arrows) == 0:
            errors.append("Mermaid code must contain at least one arrow")
        
        # Check for parse errors (common patterns)
        if re.search(r'start\(\[Start\]\)', mermaid_code):
            errors.append("Invalid node syntax: start([Start])")
        
        if re.search(r'end\(\[End\]\)', mermaid_code):
            errors.append("Invalid node syntax: end([End])")
        
        # Check for unclosed brackets
        open_brackets = mermaid_code.count('[') + mermaid_code.count('(') + mermaid_code.count('{')
        close_brackets = mermaid_code.count(']') + mermaid_code.count(')') + mermaid_code.count('}')
        if open_brackets != close_brackets:
            errors.append(f"Mismatched brackets: {open_brackets} open, {close_brackets} close")
        
        # Control flow validation
        # Check for decision branches - more thorough check
        decision_pattern = r'([A-Za-z0-9_]+)\{.*?\}'
        decisions = re.findall(decision_pattern, mermaid_code)
        for decision_id in decisions:
            # Find all edges from this decision node
            # Pattern: decision_id -->|label| target
            edge_pattern = rf'{re.escape(decision_id)}\s*-->'
            edges_from_decision = re.findall(edge_pattern, mermaid_code)
            
            if len(edges_from_decision) == 0:
                errors.append(f"Decision node {decision_id} has no outgoing edges")
            else:
                # Check for labeled branches (Yes/No/True/False)
                labeled_edges = re.findall(
                    rf'{re.escape(decision_id)}\s*-->.*?\|(Yes|No|YES|NO|True|False|TRUE|FALSE)',
                    mermaid_code
                )
                
                # Check if we have both Yes and No branches
                has_yes = any(label.lower() in ['yes', 'true'] for label in labeled_edges)
                has_no = any(label.lower() in ['no', 'false'] for label in labeled_edges)
                
                if not has_yes:
                    errors.append(f"Decision node {decision_id} missing Yes/True branch - add: {decision_id} -->|Yes| TARGET")
                if not has_no:
                    errors.append(f"Decision node {decision_id} missing No/False branch - add: {decision_id} -->|No| TARGET")
                if len(edges_from_decision) < 2:
                    errors.append(f"Decision node {decision_id} must have at least 2 outgoing edges (currently has {len(edges_from_decision)})")
        
        # Check for duplicate node definitions
        node_definitions = {}
        for line in mermaid_code.split("\n"):
            node_match = re.search(r'^([A-Za-z0-9_]+)(?:\[|\{|\(\[)', line.strip())
            if node_match:
                node_id = node_match.group(1)
                if node_id in node_definitions:
                    errors.append(f"Duplicate node definition: {node_id} defined multiple times")
                else:
                    node_definitions[node_id] = line
        
        # Check for undefined node references in edges
        defined_node_ids = set(node_definitions.keys())
        for line in mermaid_code.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            # Parse edge with label: NODE1 -->|label| NODE2
            # Or without label: NODE1 --> NODE2
            # Pattern: node_id -->|optional_label| target_node_id
            edge_match = re.search(r'^([A-Za-z0-9_]+)\s*-->(?:\|.*?\|)?\s*([A-Za-z0-9_]+)', line)
            if edge_match:
                from_node = edge_match.group(1)
                to_node = edge_match.group(2)
                
                # Skip if "from_node" or "to_node" is actually a label keyword
                label_keywords = {'Yes', 'No', 'YES', 'NO', 'True', 'False', 'TRUE', 'FALSE', 
                                 'NEXT', 'END', 'case', 'default', 'Case', 'Default'}
                
                if from_node not in defined_node_ids and from_node not in label_keywords:
                    errors.append(f"Edge from undefined node: {from_node}")
                if to_node not in defined_node_ids and to_node not in label_keywords:
                    # Check if this might be a label - look for the actual target after the label
                    # Pattern: NODE -->|label| TARGET
                    label_match = re.search(r'-->\|.*?\|([A-Za-z0-9_]+)', line)
                    if label_match:
                        actual_target = label_match.group(1)
                        if actual_target not in defined_node_ids and actual_target not in label_keywords:
                            errors.append(f"Edge to undefined node: {actual_target}")
                    else:
                        errors.append(f"Edge to undefined node: {to_node}")
        
        # Check for invalid edges from Start/End nodes
        start_nodes = re.findall(r'([A-Za-z0-9_]+)\(\[Start\]\)', mermaid_code)
        for start_id in start_nodes:
            # Check for labeled edges from Start
            invalid_edges = re.findall(rf'{re.escape(start_id)}\s*-->\|.*?\|', mermaid_code)
            if invalid_edges:
                errors.append(f"Start node {start_id} has labeled edges (Start should not have Yes/No branches)")
        
        end_nodes = re.findall(r'([A-Za-z0-9_]+)\(\[End\]\)', mermaid_code)
        for end_id in end_nodes:
            # Check for labeled edges to End
            invalid_edges = re.findall(rf'-->\|.*?\|{re.escape(end_id)}', mermaid_code)
            if invalid_edges:
                errors.append(f"End node {end_id} has labeled incoming edges (End should not have Yes/No branches)")
        
        # Structural validation
        if len(start_nodes) != 1:
            errors.append(f"Expected exactly one Start node, found {len(start_nodes)}")
        
        if len(end_nodes) != 1:
            errors.append(f"Expected exactly one End node, found {len(end_nodes)}")
        
        return len(errors) == 0, errors
