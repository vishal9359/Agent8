"""
CFG Canonicalization Stage.
Normalizes raw CFG before PseudoCodeModel generation.
"""

from typing import Dict, List, Set, Optional
from cfg_extractor import CFGNode, CFGEdge, NodeType


class CFGCanonicalizer:
    """Canonicalize CFG according to normalization rules."""
    
    def __init__(self):
        """Initialize canonicalizer."""
        pass
    
    def canonicalize(self, nodes: Dict[str, CFGNode], edges: List[CFGEdge]) -> tuple:
        """
        Canonicalize CFG.
        
        Args:
            nodes: Dictionary of CFG nodes
            edges: List of CFG edges
            
        Returns:
            Tuple of (canonical_nodes, canonical_edges)
        """
        canonical_nodes = nodes.copy()
        canonical_edges = edges.copy()
        
        # Apply canonicalization rules in order
        canonical_nodes, canonical_edges = self._merge_return_blocks(canonical_nodes, canonical_edges)
        canonical_nodes, canonical_edges = self._normalize_implicit_returns(canonical_nodes, canonical_edges)
        canonical_nodes, canonical_edges = self._normalize_switch_fallthrough(canonical_nodes, canonical_edges)
        canonical_nodes, canonical_edges = self._normalize_break_targets(canonical_nodes, canonical_edges)
        canonical_nodes, canonical_edges = self._normalize_continue_targets(canonical_nodes, canonical_edges)
        canonical_nodes, canonical_edges = self._convert_short_circuit_logic(canonical_nodes, canonical_edges)
        canonical_nodes, canonical_edges = self._remove_unreachable_blocks(canonical_nodes, canonical_edges)
        canonical_nodes, canonical_edges = self._collapse_empty_blocks(canonical_nodes, canonical_edges)
        canonical_nodes, canonical_edges = self._ensure_all_paths_reach_end(canonical_nodes, canonical_edges)
        
        return canonical_nodes, canonical_edges
    
    def _merge_return_blocks(self, nodes: Dict[str, CFGNode], edges: List[CFGEdge]) -> tuple:
        """Merge multiple return blocks into a single End node."""
        return_nodes = [nid for nid, node in nodes.items() if node.type == NodeType.RETURN]
        exit_nodes = [nid for nid, node in nodes.items() if node.type == NodeType.EXIT]
        
        if not exit_nodes:
            # Create exit node if doesn't exist
            exit_id = "exit"
            exit_node = CFGNode(id=exit_id, type=NodeType.EXIT, text="End")
            nodes[exit_id] = exit_node
            exit_nodes = [exit_id]
        
        exit_id = exit_nodes[0]
        
        # Redirect all return edges to exit
        for edge in edges:
            if edge.from_id in return_nodes:
                edge.to_id = exit_id
        
        return nodes, edges
    
    def _normalize_implicit_returns(self, nodes: Dict[str, CFGNode], edges: List[CFGEdge]) -> tuple:
        """Normalize implicit returns into explicit return nodes."""
        # Find nodes with no outgoing edges (except exit)
        exit_nodes = [nid for nid, node in nodes.items() if node.type == NodeType.EXIT]
        if not exit_nodes:
            return nodes, edges
        
        exit_id = exit_nodes[0]
        
        # Find nodes that should have returns
        nodes_with_outgoing = {edge.from_id for edge in edges}
        nodes_with_incoming_to_exit = {edge.from_id for edge in edges if edge.to_id == exit_id}
        
        # Add implicit returns for nodes that reach end without explicit return
        for nid, node in nodes.items():
            if (node.type != NodeType.EXIT and 
                node.type != NodeType.RETURN and
                nid not in nodes_with_outgoing and
                nid not in nodes_with_incoming_to_exit):
                # This node implicitly returns
                return_id = f"{nid}_implicit_return"
                return_node = CFGNode(
                    id=return_id,
                    type=NodeType.RETURN,
                    text="return"
                )
                nodes[return_id] = return_node
                edges.append(CFGEdge(from_id=nid, to_id=return_id))
                edges.append(CFGEdge(from_id=return_id, to_id=exit_id))
        
        return nodes, edges
    
    def _normalize_switch_fallthrough(self, nodes: Dict[str, CFGNode], edges: List[CFGEdge]) -> tuple:
        """Normalize switch fallthrough into explicit edges."""
        # This is handled during switch processing
        # Additional normalization: ensure break statements properly exit switch
        switch_nodes = [nid for nid, node in nodes.items() if node.type == NodeType.SWITCH]
        break_nodes = [nid for nid, node in nodes.items() if node.type == NodeType.BREAK]
        
        # Find switch exit points (merge nodes after switch)
        for switch_id in switch_nodes:
            # Find nodes reachable from switch
            switch_reachable = self._get_reachable_nodes(switch_id, edges)
            
            # Find break nodes within switch scope
            for break_id in break_nodes:
                if break_id in switch_reachable:
                    # Ensure break exits switch
                    # Find the switch merge/exit node
                    switch_exit = self._find_switch_exit(switch_id, nodes, edges)
                    if switch_exit:
                        # Remove existing edges from break
                        edges = [e for e in edges if e.from_id != break_id]
                        edges.append(CFGEdge(from_id=break_id, to_id=switch_exit))
        
        return nodes, edges
    
    def _find_switch_exit(self, switch_id: str, nodes: Dict[str, CFGNode], edges: List[CFGEdge]) -> Optional[str]:
        """Find the exit node for a switch statement."""
        # Look for merge nodes or nodes after switch
        switch_edges = [e for e in edges if e.from_id == switch_id]
        if not switch_edges:
            return None
        
        # Find common target (merge point)
        targets = [e.to_id for e in switch_edges]
        if not targets:
            return None
        
        # Return first target as switch exit
        return targets[0]
    
    def _normalize_break_targets(self, nodes: Dict[str, CFGNode], edges: List[CFGEdge]) -> tuple:
        """Normalize break targets to exit only loop or switch."""
        break_nodes = [nid for nid, node in nodes.items() if node.type == NodeType.BREAK]
        loop_nodes = [nid for nid, node in nodes.items() if node.type == NodeType.LOOP]
        switch_nodes = [nid for nid, node in nodes.items() if node.type == NodeType.SWITCH]
        
        for break_id in break_nodes:
            # Find containing loop or switch
            containing_scope = self._find_containing_scope(break_id, loop_nodes + switch_nodes, edges)
            if containing_scope:
                # Find exit of containing scope
                scope_exit = self._find_scope_exit(containing_scope, nodes, edges)
                if scope_exit:
                    # Redirect break to scope exit
                    edges = [e for e in edges if e.from_id != break_id or e.to_id == scope_exit]
                    if not any(e.from_id == break_id for e in edges):
                        edges.append(CFGEdge(from_id=break_id, to_id=scope_exit))
        
        return nodes, edges
    
    def _normalize_continue_targets(self, nodes: Dict[str, CFGNode], edges: List[CFGEdge]) -> tuple:
        """Normalize continue targets to loop condition."""
        continue_nodes = [nid for nid, node in nodes.items() if node.type == NodeType.CONTINUE]
        loop_nodes = [nid for nid, node in nodes.items() if node.type == NodeType.LOOP]
        
        for continue_id in continue_nodes:
            # Find containing loop
            containing_loop = self._find_containing_scope(continue_id, loop_nodes, edges)
            if containing_loop:
                # Redirect continue to loop condition
                edges = [e for e in edges if e.from_id != continue_id or e.to_id == containing_loop]
                if not any(e.from_id == continue_id for e in edges):
                    edges.append(CFGEdge(from_id=continue_id, to_id=containing_loop))
        
        return nodes, edges
    
    def _find_containing_scope(self, node_id: str, scope_nodes: List[str], edges: List[CFGEdge]) -> Optional[str]:
        """Find the containing scope (loop/switch) for a node."""
        # Build reverse graph
        reverse_edges = {}
        for edge in edges:
            if edge.to_id not in reverse_edges:
                reverse_edges[edge.to_id] = []
            reverse_edges[edge.to_id].append(edge.from_id)
        
        # BFS from node_id backwards to find scope
        visited = set()
        queue = [node_id]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            if current in scope_nodes:
                return current
            
            if current in reverse_edges:
                queue.extend(reverse_edges[current])
        
        return None
    
    def _find_scope_exit(self, scope_id: str, nodes: Dict[str, CFGNode], edges: List[CFGEdge]) -> Optional[str]:
        """Find the exit node of a scope."""
        # For loops, find the node after the loop condition's NO branch
        scope_edges = [e for e in edges if e.from_id == scope_id]
        for edge in scope_edges:
            if edge.label in ["NO", "False"]:
                return edge.to_id
        
        return None
    
    def _convert_short_circuit_logic(self, nodes: Dict[str, CFGNode], edges: List[CFGEdge]) -> tuple:
        """Convert short-circuit boolean logic into explicit decision nodes."""
        # This is a complex transformation
        # For now, we'll handle basic cases
        # Full implementation would require AST-level analysis
        return nodes, edges
    
    def _remove_unreachable_blocks(self, nodes: Dict[str, CFGNode], edges: List[CFGEdge]) -> tuple:
        """Remove unreachable blocks."""
        # Find entry node
        entry_nodes = [nid for nid, node in nodes.items() if node.type == NodeType.ENTRY]
        if not entry_nodes:
            return nodes, edges
        
        entry_id = entry_nodes[0]
        
        # Find all reachable nodes
        reachable = self._get_reachable_nodes(entry_id, edges)
        reachable.add(entry_id)  # Include entry itself
        
        # Remove unreachable nodes
        nodes = {nid: node for nid, node in nodes.items() if nid in reachable}
        edges = [e for e in edges if e.from_id in reachable and e.to_id in reachable]
        
        return nodes, edges
    
    def _get_reachable_nodes(self, start_id: str, edges: List[CFGEdge]) -> Set[str]:
        """Get all nodes reachable from start_id."""
        reachable = set()
        queue = [start_id]
        
        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue
            reachable.add(current)
            
            # Find outgoing edges
            for edge in edges:
                if edge.from_id == current and edge.to_id not in reachable:
                    queue.append(edge.to_id)
        
        return reachable
    
    def _collapse_empty_blocks(self, nodes: Dict[str, CFGNode], edges: List[CFGEdge]) -> tuple:
        """Collapse empty blocks."""
        # Find process nodes with empty or trivial text
        empty_nodes = []
        for nid, node in nodes.items():
            if (node.type == NodeType.PROCESS and 
                (not node.text or node.text.strip() in ["", "merge", "exit_loop", "merge_switch"])):
                empty_nodes.append(nid)
        
        # Collapse empty nodes
        for empty_id in empty_nodes:
            # Find incoming and outgoing edges
            incoming = [e for e in edges if e.to_id == empty_id]
            outgoing = [e for e in edges if e.from_id == empty_id]
            
            # Connect incoming to outgoing
            for in_edge in incoming:
                for out_edge in outgoing:
                    new_edge = CFGEdge(
                        from_id=in_edge.from_id,
                        to_id=out_edge.to_id,
                        label=out_edge.label
                    )
                    if new_edge not in edges:
                        edges.append(new_edge)
            
            # Remove node and its edges
            nodes.pop(empty_id, None)
            edges = [e for e in edges if e.from_id != empty_id and e.to_id != empty_id]
        
        return nodes, edges
    
    def _ensure_all_paths_reach_end(self, nodes: Dict[str, CFGNode], edges: List[CFGEdge]) -> tuple:
        """Ensure all paths reach End unless infinite loop exists."""
        exit_nodes = [nid for nid, node in nodes.items() if node.type == NodeType.EXIT]
        if not exit_nodes:
            # Create exit node
            exit_id = "exit"
            exit_node = CFGNode(id=exit_id, type=NodeType.EXIT, text="End")
            nodes[exit_id] = exit_node
            exit_nodes = [exit_id]
        
        exit_id = exit_nodes[0]
        
        # Find nodes without outgoing edges (except exit and return)
        nodes_with_outgoing = {edge.from_id for edge in edges}
        
        for nid, node in nodes.items():
            if (nid not in nodes_with_outgoing and 
                node.type != NodeType.EXIT and 
                node.type != NodeType.RETURN):
                # Check if it's in an infinite loop
                if not self._is_in_infinite_loop(nid, nodes, edges):
                    # Connect to exit
                    edges.append(CFGEdge(from_id=nid, to_id=exit_id))
        
        return nodes, edges
    
    def _is_in_infinite_loop(self, node_id: str, nodes: Dict[str, CFGNode], edges: List[CFGEdge]) -> bool:
        """Check if node is in an infinite loop."""
        # Simple check: if node can reach itself
        visited = set()
        queue = [node_id]
        
        while queue:
            current = queue.pop(0)
            if current == node_id and len(visited) > 0:
                return True
            if current in visited:
                continue
            visited.add(current)
            
            for edge in edges:
                if edge.from_id == current:
                    queue.append(edge.to_id)
        
        return False
