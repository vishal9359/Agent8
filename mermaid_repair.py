"""
Mermaid Repair Module.
Automatically fixes common issues in generated Mermaid code.
Works generically for any C++ project without hardcoding.
"""

import re
from typing import List, Tuple, Optional, Dict, Set


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
        
        # Fix issues in order (critical to fix duplicates first)
        mermaid_code = self._remove_duplicate_nodes(mermaid_code)
        mermaid_code = self._normalize_whitespace(mermaid_code)  # Clean up before next steps
        mermaid_code = self._remove_undefined_references(mermaid_code)
        mermaid_code = self._fix_invalid_edges(mermaid_code)
        mermaid_code = self._ensure_start_end_nodes(mermaid_code)
        mermaid_code = self._remove_duplicate_nodes(mermaid_code)  # Remove duplicates again after adding nodes
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
            if lines[0].strip() == "```" or lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)
        return code.strip()
    
    def _remove_duplicate_nodes(self, code: str) -> str:
        """Remove duplicate node definitions, keeping the best one."""
        lines = [l.strip() for l in code.split("\n") if l.strip()]
        seen_nodes = {}  # node_id -> (line_index, line_content, is_start_end_format)
        new_lines = []
        lines_to_skip = set()
        
        # First pass: identify all node definitions and mark duplicates
        for i, line in enumerate(lines):
            # Check if this is a node declaration
            # Pattern: ID[text] or ID{text} or ID([text]) or ID([Start]) or ID([End])
            node_match = re.search(r'^([A-Za-z0-9_]+)(?:\[|\{|\(\[)', line)
            if node_match:
                node_id = node_match.group(1)
                is_start_end = bool(re.search(r'\(\[(Start|End)\]\)', line))
                
                if node_id in seen_nodes:
                    # Duplicate found - decide which one to keep
                    old_index, old_line, old_is_start_end = seen_nodes[node_id]
                    
                    # Prefer Start/End format: ([Start]) or ([End])
                    if is_start_end and not old_is_start_end:
                        # New one is better - mark old one for removal
                        lines_to_skip.add(old_index)
                        seen_nodes[node_id] = (i, line, is_start_end)
                    else:
                        # Old one is better or equal - mark new one for removal
                        lines_to_skip.add(i)
                else:
                    # First time seeing this node
                    seen_nodes[node_id] = (i, line, is_start_end)
        
        # Second pass: build new lines, skipping duplicates
        for i, line in enumerate(lines):
            if i not in lines_to_skip:
                new_lines.append(line)
        
        return "\n".join(new_lines)
    
    def _remove_undefined_references(self, code: str) -> str:
        """Remove edges that reference undefined nodes."""
        lines = [l.strip() for l in code.split("\n") if l.strip()]
        
        # Collect all defined node IDs
        defined_nodes = set()
        for line in lines:
            node_match = re.search(r'^([A-Za-z0-9_]+)(?:\[|\{|\(\[)', line)
            if node_match:
                defined_nodes.add(node_match.group(1))
        
        # Keywords that are not nodes (labels, etc.)
        label_keywords = {'Yes', 'No', 'YES', 'NO', 'True', 'False', 'TRUE', 'FALSE', 
                         'NEXT', 'END', 'case', 'default', 'Case', 'Default'}
        
        # Remove edges to undefined nodes
        new_lines = []
        for line in lines:
            # Parse edge properly: NODE1 -->|label| NODE2 or NODE1 --> NODE2
            # Find the actual target node after any label
            target_node = None
            from_node = None
            
            # Pattern 1: NODE -->|label| TARGET
            edge_with_label = re.search(r'^([A-Za-z0-9_]+)\s*-->\|.*?\|([A-Za-z0-9_]+)', line)
            if edge_with_label:
                from_node = edge_with_label.group(1)
                target_node = edge_with_label.group(2)
            else:
                # Pattern 2: NODE --> TARGET (no label)
                edge_no_label = re.search(r'^([A-Za-z0-9_]+)\s*-->\s*([A-Za-z0-9_]+)', line)
                if edge_no_label:
                    from_node = edge_no_label.group(1)
                    target_node = edge_no_label.group(2)
            
            if from_node and target_node:
                # Check if target is a label keyword (not a node)
                if target_node in label_keywords:
                    # Replace label keywords with End node if appropriate
                    if target_node in ['NEXT', 'END']:
                        end_nodes = [n for n in defined_nodes if re.search(r'E\d+', n)]
                        if end_nodes:
                            line = re.sub(rf'-->\s*\|.*?\|\s*{target_node}\b', f'--> {end_nodes[0]}', line)
                            line = re.sub(rf'-->\s*{target_node}\b', f'--> {end_nodes[0]}', line)
                        else:
                            # Skip this edge if no End node exists yet
                            continue
                    else:
                        # Label keywords like Yes/No are fine in edge labels
                        new_lines.append(line)
                        continue
                
                # Check if nodes are defined
                if from_node not in defined_nodes and from_node not in label_keywords:
                    # Skip edges from undefined nodes
                    continue
                
                if target_node not in defined_nodes and target_node not in label_keywords:
                    # Skip edges to undefined nodes
                    continue
            
            new_lines.append(line)
        
        return "\n".join(new_lines)
    
    def _fix_invalid_edges(self, code: str) -> str:
        """Fix invalid edges (e.g., Start/End nodes with Yes/No branches)."""
        lines = code.split("\n")
        new_lines = []
        
        # Find Start and End nodes
        start_nodes = set()
        end_nodes = set()
        
        for line in lines:
            if re.search(r'\(\[Start\]\)', line):
                match = re.search(r'([A-Za-z0-9_]+)\(\[Start\]\)', line)
                if match:
                    start_nodes.add(match.group(1))
            if re.search(r'\(\[End\]\)', line):
                match = re.search(r'([A-Za-z0-9_]+)\(\[End\]\)', line)
                if match:
                    end_nodes.add(match.group(1))
        
        # Fix invalid edges
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for edges from Start with Yes/No labels
            for start_id in start_nodes:
                if line.startswith(start_id + " -->") and "|" in line:
                    # Remove the label, keep just the edge
                    line = re.sub(rf'{re.escape(start_id)}\s*-->\|.*?\|', f'{start_id} -->', line)
            
            # Check for edges to End with Yes/No labels
            for end_id in end_nodes:
                if f'-->|' in line and end_id in line:
                    # Check if target is End
                    if re.search(rf'-->\|.*?\|{re.escape(end_id)}', line):
                        # Remove the label
                        line = re.sub(r'-->\|.*?\|', '-->', line)
            
            new_lines.append(line)
        
        return "\n".join(new_lines)
    
    def _ensure_start_end_nodes(self, code: str) -> str:
        """Ensure Start and End nodes exist and are properly defined (exactly one each)."""
        lines = [l.strip() for l in code.split("\n") if l.strip()]
        start_nodes = []  # List of (line_index, node_id)
        end_nodes = []    # List of (line_index, node_id)
        
        # Find all Start/End node definitions
        for i, line in enumerate(lines):
            # Check for Start node: S1([Start])
            if re.search(r'\(\[Start\]\)', line):
                match = re.search(r'([A-Za-z0-9_]+)\(\[Start\]\)', line)
                if match:
                    start_nodes.append((i, match.group(1)))
            # Check for End node: E1([End])
            if re.search(r'\(\[End\]\)', line):
                match = re.search(r'([A-Za-z0-9_]+)\(\[End\]\)', line)
                if match:
                    end_nodes.append((i, match.group(1)))
        
        # Remove duplicate Start/End nodes - keep only the first of each
        lines_to_remove = set()
        if len(start_nodes) > 1:
            # Keep first, mark rest for removal
            for idx, (line_idx, _) in enumerate(start_nodes[1:], 1):
                lines_to_remove.add(start_nodes[idx][0])
        if len(end_nodes) > 1:
            # Keep first, mark rest for removal
            for idx, (line_idx, _) in enumerate(end_nodes[1:], 1):
                lines_to_remove.add(end_nodes[idx][0])
        
        # Build new lines, removing duplicates
        new_lines = [line for i, line in enumerate(lines) if i not in lines_to_remove]
        
        # Update variables
        has_start = len(start_nodes) > 0
        has_end = len(end_nodes) > 0
        start_node_id = start_nodes[0][1] if start_nodes else "S1"
        end_node_id = end_nodes[0][1] if end_nodes else "E1"
        lines = new_lines
        
        flowchart_found = False
        
        # Add Start node if missing
        if not has_start:
            first_node = self._find_first_node(lines)
            start_line = f"{start_node_id}([Start])"
            
            for i, line in enumerate(lines):
                new_lines.append(line)
                if not flowchart_found and line.strip().startswith("flowchart"):
                    flowchart_found = True
                    new_lines.append(start_line)
                    if first_node:
                        edge_exists = any(f"{start_node_id} -->" in l for l in lines)
                        if not edge_exists:
                            new_lines.append(f"{start_node_id} --> {first_node}")
                    else:
                        new_lines.append(f"{start_node_id} --> {end_node_id}")
        else:
            new_lines = list(lines)
        
        # Add End node if missing
        if not has_end:
            end_line = f"{end_node_id}([End])"
            nodes_to_end = self._find_nodes_needing_end(new_lines)
            
            if end_line not in "\n".join(new_lines):
                new_lines.append(end_line)
            
            for node_id in nodes_to_end:
                edge = f"{node_id} --> {end_node_id}"
                edge_exists = any(edge in l or (f"{node_id} -->" in l and end_node_id in l) for l in new_lines)
                if not edge_exists:
                    new_lines.append(edge)
        
        return "\n".join(new_lines)
    
    def _find_first_node(self, lines: List[str]) -> Optional[str]:
        """Find the first node in the flowchart (excluding Start/End)."""
        for line in lines:
            match = re.search(r'^([A-Za-z0-9_]+)(?:\[|\{|\(\[)', line.strip())
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
        nodes_with_incoming = set()
        
        # Collect all nodes and their edges
        for line in lines:
            node_match = re.search(r'^([A-Za-z0-9_]+)(?:\[|\{|\(\[)', line.strip())
            if node_match:
                node_id = node_match.group(1)
                if not re.search(r'Start|End', line, re.IGNORECASE):
                    all_nodes.add(node_id)
            
            edge_match = re.search(r'([A-Za-z0-9_]+)\s*-->(?:.*?\|)?\s*([A-Za-z0-9_]+)', line)
            if edge_match:
                from_node = edge_match.group(1)
                to_node = edge_match.group(2)
                nodes_with_outgoing.add(from_node)
                nodes_with_incoming.add(to_node)
        
        # Find return nodes (highest priority)
        for line in lines:
            if re.search(r'\(\[return', line, re.IGNORECASE):
                match = re.search(r'([A-Za-z0-9_]+)\(\[return', line, re.IGNORECASE)
                if match:
                    nodes_to_end.append(match.group(1))
        
        # Find nodes without outgoing edges
        for node in all_nodes:
            if (node not in nodes_with_outgoing and 
                node not in nodes_to_end):
                nodes_to_end.append(node)
        
        # Fallback
        if not nodes_to_end and all_nodes:
            process_nodes = []
            for line in lines:
                if re.search(r'\[', line) and not re.search(r'return|Start|End', line, re.IGNORECASE):
                    match = re.search(r'([A-Za-z0-9_]+)\[', line)
                    if match:
                        process_nodes.append(match.group(1))
            if process_nodes:
                nodes_to_end.append(process_nodes[-1])
            else:
                nodes_to_end.append(list(all_nodes)[0] if all_nodes else "S1")
        
        return nodes_to_end
    
    def _fix_decision_branches(self, code: str) -> str:
        """Fix decision nodes missing Yes/No branches."""
        lines = [l.strip() for l in code.split("\n") if l.strip()]
        decision_nodes = {}
        node_edges = {}
        defined_nodes = set()
        
        # Find all decision nodes and defined nodes
        for line in lines:
            match = re.search(r'^([A-Za-z0-9_]+)\{.*?\}', line)
            if match:
                decision_id = match.group(1)
                decision_nodes[decision_id] = line
                defined_nodes.add(decision_id)
            
            # Also collect other node types
            node_match = re.search(r'^([A-Za-z0-9_]+)(?:\[|\(\[)', line)
            if node_match:
                defined_nodes.add(node_match.group(1))
        
        # Find all edges from each decision node
        for line in lines:
            edge_match = re.search(r'^([A-Za-z0-9_]+)\s*-->(.*)', line)
            if edge_match:
                from_node = edge_match.group(1)
                if from_node in decision_nodes:
                    if from_node not in node_edges:
                        node_edges[from_node] = []
                    node_edges[from_node].append(line)
        
        # Fix decision nodes missing branches
        new_lines = []
        nodes_added = set()
        end_node_id = "E1"
        
        # Find End node
        for line in lines:
            if re.search(r'\(\[End\]\)', line):
                match = re.search(r'([A-Za-z0-9_]+)\(\[End\]\)', line)
                if match:
                    end_node_id = match.group(1)
                    break
        
        for line in lines:
            new_lines.append(line)
            
            # Check if this is a decision node declaration
            match = re.search(r'^([A-Za-z0-9_]+)\{.*?\}', line)
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
                        if target_match and target_match.group(1) in defined_nodes:
                            yes_target = target_match.group(1)
                    elif re.search(r'\|(No|NO|False|FALSE)\|', edge):
                        has_no = True
                        target_match = re.search(r'-->\|.*?\|([A-Za-z0-9_]+)', edge)
                        if target_match and target_match.group(1) in defined_nodes:
                            no_target = target_match.group(1)
                
                # Fix missing branches
                if decision_id not in nodes_added:
                    nodes_added.add(decision_id)
                    
                    # Find valid target for missing branches
                    if not no_target:
                        next_node = self._find_next_valid_node_after(lines, line, defined_nodes, end_node_id)
                        if not has_no:
                            new_lines.append(f"{decision_id} -->|No| {next_node}")
                            has_no = True
                            no_target = next_node
                    
                    if not has_yes:
                        yes_target = self._find_yes_branch_target(lines, decision_id, no_target, defined_nodes)
                        if not yes_target:
                            yes_target = no_target
                        new_lines.append(f"{decision_id} -->|Yes| {yes_target}")
        
        return "\n".join(new_lines)
    
    def _find_next_valid_node_after(self, lines: List[str], current_line: str, 
                                    defined_nodes: Set[str], end_node_id: str) -> str:
        """Find the next valid node after current line."""
        current_index = -1
        for i, line in enumerate(lines):
            if line == current_line:
                current_index = i
                break
        
        if current_index >= 0:
            for i in range(current_index + 1, len(lines)):
                match = re.search(r'^([A-Za-z0-9_]+)(?:\[|\{|\(\[)', lines[i].strip())
                if match:
                    node_id = match.group(1)
                    if node_id in defined_nodes and not re.search(r'Start|End', lines[i], re.IGNORECASE):
                        return node_id
        
        return end_node_id
    
    def _find_yes_branch_target(self, lines: List[str], decision_id: str, no_target: str,
                                defined_nodes: Set[str]) -> Optional[str]:
        """Find target for Yes branch of a decision node."""
        # Look for unlabeled edge from this decision
        for line in lines:
            if line.strip().startswith(decision_id + " -->") and "|" not in line:
                target_match = re.search(r'-->\s*([A-Za-z0-9_]+)', line)
                if target_match and target_match.group(1) in defined_nodes:
                    return target_match.group(1)
        
        # Look for first process node after decision
        for line in lines:
            if decision_id in line:
                continue
            match = re.search(r'^([A-Za-z0-9_]+)\[', line.strip())
            if match and match.group(1) in defined_nodes:
                return match.group(1)
        
        return no_target if no_target in defined_nodes else None
    
    def _fix_node_syntax(self, code: str) -> str:
        """Fix common node syntax errors."""
        lines = code.split("\n")
        new_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Fix invalid syntax
            line = re.sub(r'start\(\[Start\]\)', 'S1([Start])', line, flags=re.IGNORECASE)
            line = re.sub(r'end\(\[End\]\)', 'E1([End])', line, flags=re.IGNORECASE)
            new_lines.append(line)
        
        return "\n".join(new_lines)
    
    def _ensure_connectivity(self, code: str) -> str:
        """Ensure all nodes are connected properly."""
        lines = [l.strip() for l in code.split("\n") if l.strip()]
        all_nodes = set()
        nodes_with_incoming = set()
        nodes_with_outgoing = set()
        start_node_id = "S1"
        end_node_id = "E1"
        
        # Collect all nodes
        for line in lines:
            match = re.search(r'^([A-Za-z0-9_]+)(?:\[|\{|\(\[)', line)
            if match:
                all_nodes.add(match.group(1))
                if re.search(r'\(\[Start\]\)', line):
                    start_node_id = match.group(1)
                if re.search(r'\(\[End\]\)', line):
                    end_node_id = match.group(1)
            
            # Track edges
            edge_match = re.search(r'([A-Za-z0-9_]+)\s*-->(?:.*?\|)?\s*([A-Za-z0-9_]+)', line)
            if edge_match:
                from_node = edge_match.group(1)
                to_node = edge_match.group(2)
                if from_node in all_nodes and to_node in all_nodes:
                    nodes_with_outgoing.add(from_node)
                    nodes_with_incoming.add(to_node)
        
        # Ensure Start has outgoing edge
        if start_node_id in all_nodes and start_node_id not in nodes_with_outgoing:
            first_node = self._find_first_node(lines)
            if first_node and first_node in all_nodes:
                lines.append(f"{start_node_id} --> {first_node}")
        
        return "\n".join(lines)
    
    def _normalize_whitespace(self, code: str) -> str:
        """Normalize whitespace in Mermaid code."""
        lines = []
        for line in code.split("\n"):
            line = line.strip()
            if line:
                lines.append(line)
        return "\n".join(lines)
    
    def _generate_empty_flowchart(self) -> str:
        """Generate a minimal valid flowchart."""
        return """flowchart TD
    S1([Start])
    E1([End])
    S1 --> E1"""
