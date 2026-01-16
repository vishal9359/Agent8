"""
PseudoCodeModel IR Generator.
Converts canonicalized CFG to LLM-optimized JSON IR.
"""

from typing import Dict, List, Optional
from cfg_extractor import CFGNode, CFGEdge, NodeType
import json


class PseudoCodeModel:
    """PseudoCodeModel Intermediate Representation."""
    
    def __init__(self, entry_function: str, file: str, description: str = ""):
        """
        Initialize PseudoCodeModel.
        
        Args:
            entry_function: Function name
            file: Source file path
            description: Function description
        """
        self.entry_function = entry_function
        self.file = file
        self.description = description
        self.steps: List[Dict] = []
        self.edges: List[Dict] = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "entry_function": self.entry_function,
            "file": self.file,
            "description": self.description,
            "steps": self.steps,
            "edges": self.edges
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_cfg(cls, nodes: Dict[str, CFGNode], edges: List[CFGEdge], 
                 entry_function: str, file: str, description: str = "") -> 'PseudoCodeModel':
        """
        Generate PseudoCodeModel from canonicalized CFG.
        
        Args:
            nodes: Canonicalized CFG nodes
            edges: Canonicalized CFG edges
            entry_function: Function name
            file: Source file path
            description: Function description
            
        Returns:
            PseudoCodeModel instance
        """
        model = cls(entry_function, file, description)
        
        # Map CFG node types to IR step types
        type_mapping = {
            NodeType.ENTRY: "start",
            NodeType.EXIT: "end",
            NodeType.PROCESS: "process",
            NodeType.DECISION: "decision",
            NodeType.LOOP: "loop",
            NodeType.SWITCH: "switch",
            NodeType.CASE: "case",
            NodeType.DEFAULT: "default",
            NodeType.BREAK: "break",
            NodeType.CONTINUE: "continue",
            NodeType.RETURN: "return",
            NodeType.THROW: "throw"
        }
        
        # Convert nodes to steps
        for node_id, node in nodes.items():
            step_type = type_mapping.get(node.type, "process")
            
            # Clean up text
            text = node.text.strip()
            if not text or text in ["merge", "exit_loop", "merge_switch"]:
                if step_type == "start":
                    text = "Start"
                elif step_type == "end":
                    text = "End"
                else:
                    text = step_type
            
            step = {
                "id": node_id,
                "type": step_type,
                "text": text
            }
            model.steps.append(step)
        
        # Convert edges
        for edge in edges:
            edge_dict = {
                "from": edge.from_id,
                "to": edge.to_id
            }
            if edge.label:
                edge_dict["label"] = edge.label
            model.edges.append(edge_dict)
        
        return model
    
    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate PseudoCodeModel.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check for start node
        start_nodes = [s for s in self.steps if s["type"] == "start"]
        if len(start_nodes) != 1:
            errors.append(f"Expected exactly one start node, found {len(start_nodes)}")
        
        # Check for end node
        end_nodes = [s for s in self.steps if s["type"] == "end"]
        if len(end_nodes) != 1:
            errors.append(f"Expected exactly one end node, found {len(end_nodes)}")
        
        # Check step IDs are unique
        step_ids = [s["id"] for s in self.steps]
        if len(step_ids) != len(set(step_ids)):
            errors.append("Duplicate step IDs found")
        
        # Check edge references
        valid_ids = set(step_ids)
        for edge in self.edges:
            if edge["from"] not in valid_ids:
                errors.append(f"Edge references unknown 'from' node: {edge['from']}")
            if edge["to"] not in valid_ids:
                errors.append(f"Edge references unknown 'to' node: {edge['to']}")
        
        # Check decision nodes have labels
        decision_ids = {s["id"] for s in self.steps if s["type"] == "decision"}
        for edge in self.edges:
            if edge["from"] in decision_ids:
                if "label" not in edge:
                    errors.append(f"Decision node {edge['from']} edge missing label")
        
        # Check loop nodes
        loop_ids = {s["id"] for s in self.steps if s["type"] == "loop"}
        for loop_id in loop_ids:
            loop_edges = [e for e in self.edges if e["from"] == loop_id]
            if len(loop_edges) < 2:
                errors.append(f"Loop node {loop_id} should have at least 2 outgoing edges")
        
        return len(errors) == 0, errors
