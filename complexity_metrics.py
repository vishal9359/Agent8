"""
Complexity Metrics Calculator.
Computes cyclomatic complexity and other metrics from CFG.
"""

from typing import Dict, List
from cfg_extractor import CFGNode, CFGEdge, NodeType


class ComplexityMetrics:
    """Calculate complexity metrics from CFG."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        pass
    
    def calculate(self, nodes: Dict[str, CFGNode], edges: List[CFGEdge], 
                  ir_model: Dict = None) -> Dict:
        """
        Calculate complexity metrics.
        
        Args:
            nodes: CFG nodes
            edges: CFG edges
            ir_model: Optional PseudoCodeModel for additional metrics
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic counts
        metrics["node_count"] = len(nodes)
        metrics["edge_count"] = len(edges)
        
        # Decision count
        decision_count = sum(1 for node in nodes.values() if node.type == NodeType.DECISION)
        metrics["decision_count"] = decision_count
        
        # Loop count
        loop_count = sum(1 for node in nodes.values() if node.type == NodeType.LOOP)
        metrics["loop_count"] = loop_count
        
        # Exception paths
        exception_count = sum(1 for node in nodes.values() if node.type == NodeType.THROW)
        metrics["exception_paths"] = exception_count
        
        # Cyclomatic complexity: E - N + 2P
        # E = number of edges
        # N = number of nodes
        # P = number of connected components (always 1 for single function)
        E = len(edges)
        N = len(nodes)
        P = 1  # Single function
        
        cyclomatic_complexity = E - N + 2 * P
        metrics["cyclomatic_complexity"] = cyclomatic_complexity
        
        # Max depth (longest path from entry to exit)
        entry_nodes = [nid for nid, node in nodes.items() if node.type == NodeType.ENTRY]
        exit_nodes = [nid for nid, node in nodes.items() if node.type == NodeType.EXIT]
        
        if entry_nodes and exit_nodes:
            entry_id = entry_nodes[0]
            exit_id = exit_nodes[0]
            max_depth = self._calculate_max_depth(entry_id, exit_id, nodes, edges)
            metrics["max_depth"] = max_depth
        else:
            metrics["max_depth"] = 0
        
        # Additional metrics from IR if available
        if ir_model:
            metrics["ir_step_count"] = len(ir_model.get("steps", []))
            metrics["ir_edge_count"] = len(ir_model.get("edges", []))
        
        return metrics
    
    def _calculate_max_depth(self, start_id: str, end_id: str, 
                            nodes: Dict[str, CFGNode], edges: List[CFGEdge]) -> int:
        """Calculate maximum depth (longest path) from start to end."""
        # Build adjacency list
        graph = {}
        for edge in edges:
            if edge.from_id not in graph:
                graph[edge.from_id] = []
            graph[edge.from_id].append(edge.to_id)
        
        # DFS to find longest path
        visited = set()
        max_depth = 0
        
        def dfs(node_id: str, depth: int):
            nonlocal max_depth
            if node_id == end_id:
                max_depth = max(max_depth, depth)
                return
            
            if node_id in visited:
                return
            
            visited.add(node_id)
            
            if node_id in graph:
                for neighbor in graph[node_id]:
                    dfs(neighbor, depth + 1)
            
            visited.remove(node_id)
        
        dfs(start_id, 0)
        return max_depth
