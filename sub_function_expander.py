"""
Sub-Function Expansion Module.
Expands allowed sub-functions with depth limit and cycle detection.
"""

from typing import Dict, List, Set, Optional
from ast_parser import ASTParser
from cfg_extractor import CFGExtractor
from cfg_canonicalizer import CFGCanonicalizer
from pseudo_code_model import PseudoCodeModel


class SubFunctionExpander:
    """Expand sub-functions in CFG."""
    
    def __init__(self, allowed_functions: List[str] = None, max_depth: int = 1):
        """
        Initialize sub-function expander.
        
        Args:
            allowed_functions: List of function names to expand
            max_depth: Maximum expansion depth (default: 1)
        """
        self.allowed_functions = allowed_functions or []
        self.max_depth = max_depth
        self.ast_parser = ASTParser()
        self.cfg_extractor = CFGExtractor()
        self.cfg_canonicalizer = CFGCanonicalizer()
        self.expanded_functions: Set[str] = set()
        self.expansion_depth: Dict[str, int] = {}
    
    def expand(self, ir_model: Dict, ast_data: Dict, 
               function_name: str) -> Dict:
        """
        Expand sub-functions in IR model.
        
        Args:
            ir_model: PseudoCodeModel dictionary
            ast_data: AST data from parser
            function_name: Current function name
            
        Returns:
            Expanded IR model
        """
        if not self.allowed_functions:
            return ir_model
        
        # Reset expansion tracking
        self.expanded_functions = {function_name}
        self.expansion_depth = {function_name: 0}
        
        # Find function calls in IR
        expanded_ir = self._expand_function_calls(ir_model, ast_data, function_name, 0)
        
        return expanded_ir
    
    def _expand_function_calls(self, ir_model: Dict, ast_data: Dict, 
                               current_function: str, depth: int) -> Dict:
        """Recursively expand function calls."""
        if depth >= self.max_depth:
            return ir_model
        
        # Find process nodes that might be function calls
        expanded_steps = []
        expanded_edges = []
        
        for step in ir_model["steps"]:
            if step["type"] == "process":
                # Check if this is a function call
                func_name = self._extract_function_name(step["text"])
                
                if func_name and func_name in self.allowed_functions:
                    # Check for cycles
                    if func_name in self.expanded_functions:
                        # Cycle detected, keep as atomic node
                        expanded_steps.append(step)
                        continue
                    
                    # Check depth
                    if self.expansion_depth.get(func_name, 0) >= self.max_depth:
                        expanded_steps.append(step)
                        continue
                    
                    # Expand function
                    try:
                        func_node = self.ast_parser.find_function(ast_data, func_name)
                        if func_node:
                            func_body = self.ast_parser.get_function_body(
                                func_node, ast_data["source"]
                            )
                            
                            if func_body:
                                # Extract CFG
                                nodes, edges = self.cfg_extractor.extract(
                                    func_body, ast_data["source"]
                                )
                                
                                # Canonicalize
                                nodes, edges = self.cfg_canonicalizer.canonicalize(nodes, edges)
                                
                                # Generate IR for sub-function
                                sub_ir = PseudoCodeModel.from_cfg(
                                    nodes, edges, func_name, 
                                    ast_data.get("file_path", ""), 
                                    f"Sub-function: {func_name}"
                                )
                                
                                # Mark as expanded
                                self.expanded_functions.add(func_name)
                                self.expansion_depth[func_name] = depth + 1
                                
                                # Recursively expand sub-function
                                sub_ir_dict = self._expand_function_calls(
                                    sub_ir.to_dict(), ast_data, func_name, depth + 1
                                )
                                
                                # Merge sub-function IR into main IR
                                # Replace current step with sub-function steps
                                # Adjust IDs to avoid conflicts
                                id_prefix = f"{step['id']}_sub_"
                                
                                for sub_step in sub_ir_dict["steps"]:
                                    new_step = sub_step.copy()
                                    new_step["id"] = id_prefix + sub_step["id"]
                                    expanded_steps.append(new_step)
                                
                                # Add edges from sub-function
                                for sub_edge in sub_ir_dict["edges"]:
                                    new_edge = sub_edge.copy()
                                    new_edge["from"] = id_prefix + sub_edge["from"]
                                    new_edge["to"] = id_prefix + sub_edge["to"]
                                    expanded_edges.append(new_edge)
                                
                                # Connect: replace step with sub-function entry
                                sub_start = [s for s in sub_ir_dict["steps"] if s["type"] == "start"][0]
                                sub_end = [s for s in sub_ir_dict["steps"] if s["type"] == "end"][0]
                                
                                # Redirect edges to/from this step
                                for edge in ir_model["edges"]:
                                    if edge["from"] == step["id"]:
                                        new_edge = edge.copy()
                                        new_edge["from"] = id_prefix + sub_start["id"]
                                        expanded_edges.append(new_edge)
                                    elif edge["to"] == step["id"]:
                                        new_edge = edge.copy()
                                        new_edge["to"] = id_prefix + sub_end["id"]
                                        expanded_edges.append(new_edge)
                                
                                continue
                    except Exception as e:
                        # Expansion failed, keep as atomic
                        pass
                
            expanded_steps.append(step)
        
        # Add non-expanded edges
        for edge in ir_model["edges"]:
            # Only add if not replaced by expansion
            from_expanded = any(
                edge["from"] == step["id"] and step["type"] == "process" 
                and self._extract_function_name(step["text"]) in self.allowed_functions
                for step in ir_model["steps"]
            )
            to_expanded = any(
                edge["to"] == step["id"] and step["type"] == "process"
                and self._extract_function_name(step["text"]) in self.allowed_functions
                for step in ir_model["steps"]
            )
            
            if not (from_expanded or to_expanded):
                expanded_edges.append(edge)
        
        return {
            "entry_function": ir_model["entry_function"],
            "file": ir_model["file"],
            "description": ir_model.get("description", ""),
            "steps": expanded_steps,
            "edges": expanded_edges
        }
    
    def _extract_function_name(self, text: str) -> Optional[str]:
        """Extract function name from text."""
        # Simple heuristic: look for function call pattern
        text = text.strip()
        
        # Pattern: identifier(...)
        import re
        match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', text)
        if match:
            return match.group(1)
        
        return None
