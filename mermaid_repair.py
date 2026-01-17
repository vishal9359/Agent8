"""
Mermaid Repair Module.
Automatically fixes common issues in generated Mermaid code.
Works generically for any C++ project without hardcoding.
"""

import re
from typing import List, Tuple, Optional


class MermaidRepair:
    """Repair and fix common issues in Mermaid flowcharts."""
    
    def __init__(self):
        """Initialize Mermaid repair module."""
        pass
    
    def repair(self, mermaid_code: str) -> str:
        """
        Repair Mermaid code by fixing common issues.
        
        Args:
            mermaid_code: Potentially broken Mermaid code
            
        Returns:
            Repaired Mermaid code
        """
        if not mermaid_code or not mermaid_code.strip():
            return self._generate_empty_flowchart()
        
        # Remove markdown code blocks if present
        mermaid_code = self._clean_markdown(mermaid_code)
        
        # Ensure it starts with flowchart TD
        if not mermaid_code.strip().startswith("flowchart"):
            mermaid_code = "flowchart TD\n" + mermaid_code
        
        # Fix common issues
        mermaid_code = self._ensure_start_end_nodes(mermaid_code)
        mermaid_code = self._fix_decision_branches(mermaid_code)
        mermaid_code = self._fix_node_syntax(mermaid_code)
        mermaid_code = self._ensure_connectivity(mermaid_code)
        mermaid_code = self._normalize_whitespace(mermaid_code)
        
        return mermaid_code
    
    def _clean_markdown(self, code: str) -> str:
        """Remove markdown code blocks."""
        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            # Remove first line if it's ```
            if lines[0].strip() == "```" or lines[0].strip().startswith("```"):
                lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)
        return code.strip()
    
    def _ensure_start_end_nodes(self, code: str) -> str:
        """Ensure Start and End nodes exist."""
        lines = code.split("\n")
        has_start = False
        has_end = False
        start_node_id = None
        end_node_id = None
        
        # Find existing Start/End nodes
        for line in lines:
            if re.search(r'\(\[Start\]\)', line):
                has_start = True
                match = re.search(r'([A-Za-z0-9_]+)\(\[Start\]\)', line)
                if match:
                    start_node_id = match.group(1)
            if re.search(r'\(\[End\]\)', line):
                has_end = True
                match = re.search(r'([A-Za-z0-9_]+)\(\[End\]\)', line)
                if match:
                    end_node_id = match.group(1)
        
        # Add Start node if missing
        if not has_start:
            if not start_node_id:
                start_node_id = "S1"
            # Find first node that's not Start/End to connect from
            first_node = self._find_first_node(lines)
            start_line = f"{start_node_id}([Start])"
            if start_line not in code:
                # Insert after flowchart TD line
                new_lines = []
                for i, line in enumerate(lines):
                    new_lines.append(line)
                    if i == 0 and line.strip().startswith("flowchart"):
                        new_lines.append(start_line)
                        if first_node:
                            new_lines.append(f"{start_node_id} --> {first_node}")
                lines = new_lines
                has_start = True
        
        # Add End node if missing
        if not has_end:
            if not end_node_id:
                end_node_id = "E1"
            # Find nodes that should connect to End (returns, final processes)
            end_line = f"{end_node_id}([End])"
            if end_line not in code:
                # Find nodes that need to connect to End
                nodes_to_end = self._find_nodes_needing_end(lines)
                new_lines = list(lines)
                new_lines.append(end_line)
                for node_id in nodes_to_end:
                    edge = f"{node_id} --> {end_node_id}"
                    if edge not in code:
                        new_lines.append(edge)
                lines = new_lines
                has_end = True
        
        return "\n".join(lines)
    
    def _find_first_node(self, lines: List[str]) -> Optional[str]:
        """Find the first node in the flowchart (excluding Start/End)."""
        for line in lines:
            # Look for node declarations: ID[text] or ID{text} or ID([text])
            match = re.search(r'([A-Za-z0-9_]+)(?:\[|\{|\(\[)', line)
            if match:
                node_id = match.group(1)
                if not re.search(r'Start|End', line, re.IGNORECASE):
                    return node_id
        return None
    
    def _find_nodes_needing_end(self, lines: List[str]) -> List[str]:
        """Find nodes that should connect to End (returns, final nodes)."""
        nodes_to_end = []
        all_nodes = set()
        nodes_with_outgoing = set()
        
        # Collect all nodes and their outgoing edges
        for line in lines:
            # Find node declarations
            node_match = re.search(r'([A-Za-z0-9_]+)(?:\[|\{|\(\[)', line)
            if node_match:
                all_nodes.add(node_match.group(1))
            
            # Find edges: node1 --> node2
            edge_match = re.search(r'([A-Za-z0-9_]+)\s*-->', line)
            if edge_match:
                nodes_with_outgoing.add(edge_match.group(1))
        
        # Find return nodes
        for line in lines:
            if re.search(r'\(\[return', line, re.IGNORECASE):
                match = re.search(r'([A-Za-z0-9_]+)\(\[return', line, re.IGNORECASE)
                if match:
                    nodes_to_end.append(match.group(1))
        
        # Find nodes without outgoing edges (except Start/End)
        for node in all_nodes:
            if (node not in nodes_with_outgoing and 
                not re.search(r'Start|End', node, re.IGNORECASE) and
                node not in nodes_to_end):
                nodes_to_end.append(node)
        
        return nodes_to_end if nodes_to_end else list(all_nodes - nodes_with_outgoing)[:1]
    
    def _fix_decision_branches(self, code: str) -> str:
        """Fix decision nodes missing Yes/No branches."""
        lines = code.split("\n")
        decision_nodes = {}
        node_edges = {}
        
        # Find all decision nodes
        for line in lines:
            match = re.search(r'([A-Za-z0-9_]+)\{.*?\}', line)
            if match:
                decision_id = match.group(1)
                decision_nodes[decision_id] = line
        
        # Find all edges from each decision node
        for line in lines:
            edge_match = re.search(r'([A-Za-z0-9_]+)\s*-->(.*)', line)
            if edge_match:
                from_node = edge_match.group(1)
                if from_node in decision_nodes:
                    if from_node not in node_edges:
                        node_edges[from_node] = []
                    node_edges[from_node].append(line)
        
        # Fix decision nodes missing branches
        new_lines = []
        nodes_added = set()
        
        for line in lines:
            new_lines.append(line)
            
            # Check if this is a decision node declaration
            match = re.search(r'([A-Za-z0-9_]+)\{.*?\}', line)
            if match:
                decision_id = match.group(1)
                
                # Check if it has proper branches
                edges = node_edges.get(decision_id, [])
                has_yes = False
                has_no = False
                yes_target = None
                no_target = None
                
                for edge in edges:
                    if re.search(r'\|(Yes|YES|True|TRUE)\|', edge):
                        has_yes = True
                        target_match = re.search(r'-->\|.*?\|([A-Za-z0-9_]+)', edge)
                        if target_match:
                            yes_target = target_match.group(1)
                    elif re.search(r'\|(No|NO|False|FALSE)\|', edge):
                        has_no = True
                        target_match = re.search(r'-->\|.*?\|([A-Za-z0-9_]+)', edge)
                        if target_match:
                            no_target = target_match.group(1)
                
                # Fix missing branches
                if decision_id not in nodes_added:
                    nodes_added.add(decision_id)
                    
                    # Find next node after this decision (for No branch if missing)
                    if not no_target:
                        # Look for next node in the code
                        next_node = self._find_next_node_after(lines, line)
                        if not next_node:
                            # Use End node
                            end_match = re.search(r'([A-Za-z0-9_]+)\(\[End\]\)', code)
                            if end_match:
                                next_node = end_match.group(1)
                            else:
                                next_node = "E1"
                        
                        if not has_no:
                            new_lines.append(f"{decision_id} -->|No| {next_node}")
                            has_no = True
                            no_target = next_node
                    
                    # For Yes branch, find the first node after decision in original flow
                    if not has_yes:
                        # Look for unlabeled edge or first process after decision
                        yes_target = self._find_yes_branch_target(lines, decision_id, no_target)
                        if not yes_target:
                            yes_target = no_target  # Fallback
                        new_lines.append(f"{decision_id} -->|Yes| {yes_target}")
        
        return "\n".join(new_lines)
    
    def _find_next_node_after(self, lines: List[str], current_line: str) -> Optional[str]:
        """Find the next node declaration after current line."""
        current_index = -1
        for i, line in enumerate(lines):
            if line == current_line:
                current_index = i
                break
        
        if current_index >= 0:
            for i in range(current_index + 1, len(lines)):
                match = re.search(r'([A-Za-z0-9_]+)(?:\[|\{|\(\[)', lines[i])
                if match:
                    node_id = match.group(1)
                    if not re.search(r'Start|End', lines[i], re.IGNORECASE):
                        return node_id
        
        return None
    
    def _find_yes_branch_target(self, lines: List[str], decision_id: str, no_target: str) -> Optional[str]:
        """Find target for Yes branch of a decision node."""
        # Look for unlabeled edge from this decision
        for line in lines:
            if line.strip().startswith(decision_id + " -->"):
                # Check if it's unlabeled
                if "|" not in line:
                    target_match = re.search(r'-->\s*([A-Za-z0-9_]+)', line)
                    if target_match:
                        return target_match.group(1)
        
        # Look for first process node after decision
        return self._find_next_node_after(lines, f"{decision_id}{{")
    
    def _fix_node_syntax(self, code: str) -> str:
        """Fix common node syntax errors."""
        lines = code.split("\n")
        new_lines = []
        
        for line in lines:
            # Fix invalid syntax like start([Start])
            line = re.sub(r'start\(\[Start\]\)', 'S1([Start])', line, flags=re.IGNORECASE)
            line = re.sub(r'end\(\[End\]\)', 'E1([End])', line, flags=re.IGNORECASE)
            
            new_lines.append(line)
        
        return "\n".join(new_lines)
    
    def _ensure_connectivity(self, code: str) -> str:
        """Ensure all nodes are connected."""
        lines = code.split("\n")
        all_nodes = set()
        nodes_with_incoming = set()
        nodes_with_outgoing = set()
        
        # Collect all nodes
        for line in lines:
            match = re.search(r'([A-Za-z0-9_]+)(?:\[|\{|\(\[)', line)
            if match:
                all_nodes.add(match.group(1))
            
            # Track edges
            edge_match = re.search(r'([A-Za-z0-9_]+)\s*-->\s*([A-Za-z0-9_]+)', line)
            if edge_match:
                nodes_with_outgoing.add(edge_match.group(1))
                nodes_with_outgoing.add(edge_match.group(2))  # Also mark target as having incoming
                nodes_with_incoming.add(edge_match.group(2))
        
        # Ensure Start has outgoing edge
        start_nodes = [n for n in all_nodes if re.search(r'S\d+', n)]
        if start_nodes and start_nodes[0] not in nodes_with_outgoing:
            first_node = self._find_first_node(lines)
            if first_node:
                lines.append(f"{start_nodes[0]} --> {first_node}")
        
        return "\n".join(lines)
    
    def _normalize_whitespace(self, code: str) -> str:
        """Normalize whitespace in Mermaid code."""
        lines = []
        for line in code.split("\n"):
            line = line.strip()
            if line:  # Skip empty lines
                lines.append(line)
        return "\n".join(lines)
    
    def _generate_empty_flowchart(self) -> str:
        """Generate a minimal valid flowchart."""
        return """flowchart TD
    S1([Start])
    E1([End])
    S1 --> E1"""
