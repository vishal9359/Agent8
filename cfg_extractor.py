"""
Control Flow Graph (CFG) extraction from C++ AST.
Extracts raw CFG before canonicalization.
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class NodeType(Enum):
    """CFG node types."""
    ENTRY = "entry"
    EXIT = "exit"
    PROCESS = "process"
    DECISION = "decision"
    LOOP = "loop"
    SWITCH = "switch"
    CASE = "case"
    DEFAULT = "default"
    BREAK = "break"
    CONTINUE = "continue"
    RETURN = "return"
    THROW = "throw"


@dataclass
class CFGNode:
    """CFG node representation."""
    id: str
    type: NodeType
    text: str
    ast_node: Optional[object] = None
    start_byte: int = 0
    end_byte: int = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class CFGEdge:
    """CFG edge representation."""
    from_id: str
    to_id: str
    label: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class CFGExtractor:
    """Extract Control Flow Graph from C++ AST."""
    
    def __init__(self):
        """Initialize CFG extractor."""
        self.node_counter = 0
        self.nodes: Dict[str, CFGNode] = {}
        self.edges: List[CFGEdge] = []
        self.source_code: str = ""
    
    def _new_node_id(self) -> str:
        """Generate unique node ID."""
        self.node_counter += 1
        return f"n{self.node_counter}"
    
    def extract(self, body_node: dict, source_code: str) -> Tuple[Dict[str, CFGNode], List[CFGEdge]]:
        """
        Extract CFG from function body AST node.
        
        Args:
            body_node: Function body node dictionary
            source_code: Source code string
            
        Returns:
            Tuple of (nodes dict, edges list)
        """
        self.source_code = source_code
        self.nodes = {}
        self.edges = []
        self.node_counter = 0
        
        # Create entry node
        entry_id = self._new_node_id()
        entry_node = CFGNode(
            id=entry_id,
            type=NodeType.ENTRY,
            text="Start"
        )
        self.nodes[entry_id] = entry_node
        
        # Process compound statement
        compound_node = body_node['node']
        last_id = self._process_statement_list(compound_node, entry_id)
        
        # Create exit node
        exit_id = self._new_node_id()
        exit_node = CFGNode(
            id=exit_id,
            type=NodeType.EXIT,
            text="End"
        )
        self.nodes[exit_id] = exit_node
        
        # Connect last node to exit if no return
        if last_id and last_id != exit_id:
            self.edges.append(CFGEdge(from_id=last_id, to_id=exit_id))
        
        return self.nodes, self.edges
    
    def _process_statement_list(self, compound_node, prev_id: str) -> Optional[str]:
        """Process statement list in compound statement."""
        statements = []
        
        # Extract statements from compound statement
        for child in compound_node.children:
            if child.type in ['declaration', 'expression_statement', 'return_statement',
                             'if_statement', 'while_statement', 'for_statement',
                             'do_statement', 'switch_statement', 'break_statement',
                             'continue_statement', 'compound_statement']:
                statements.append(child)
        
        current_id = prev_id
        
        for stmt in statements:
            current_id = self._process_statement(stmt, current_id)
            if current_id is None:
                break
        
        return current_id
    
    def _process_statement(self, stmt_node, prev_id: str) -> Optional[str]:
        """Process individual statement."""
        stmt_type = stmt_node.type
        
        if stmt_type == 'return_statement':
            return self._process_return(stmt_node, prev_id)
        elif stmt_type == 'if_statement':
            return self._process_if(stmt_node, prev_id)
        elif stmt_type == 'while_statement':
            return self._process_while(stmt_node, prev_id)
        elif stmt_type == 'for_statement':
            return self._process_for(stmt_node, prev_id)
        elif stmt_type == 'do_statement':
            return self._process_do_while(stmt_node, prev_id)
        elif stmt_type == 'switch_statement':
            return self._process_switch(stmt_node, prev_id)
        elif stmt_type == 'break_statement':
            return self._process_break(stmt_node, prev_id)
        elif stmt_type == 'continue_statement':
            return self._process_continue(stmt_node, prev_id)
        elif stmt_type == 'expression_statement':
            return self._process_expression(stmt_node, prev_id)
        elif stmt_type == 'declaration':
            return self._process_declaration(stmt_node, prev_id)
        elif stmt_type == 'compound_statement':
            return self._process_statement_list(stmt_node, prev_id)
        else:
            # Default: process as expression
            return self._process_expression(stmt_node, prev_id)
    
    def _process_return(self, stmt_node, prev_id: str) -> str:
        """Process return statement."""
        node_id = self._new_node_id()
        text = self.source_code[stmt_node.start_byte:stmt_node.end_byte].strip()
        
        node = CFGNode(
            id=node_id,
            type=NodeType.RETURN,
            text=text,
            ast_node=stmt_node,
            start_byte=stmt_node.start_byte,
            end_byte=stmt_node.end_byte
        )
        self.nodes[node_id] = node
        self.edges.append(CFGEdge(from_id=prev_id, to_id=node_id))
        
        return node_id
    
    def _process_if(self, stmt_node, prev_id: str) -> str:
        """Process if statement."""
        # Extract condition
        condition_node = None
        then_node = None
        else_node = None
        
        for child in stmt_node.children:
            if child.type == 'parenthesized_expression':
                condition_node = child
            elif child.type == 'compound_statement' or child.type in ['expression_statement', 'if_statement']:
                if then_node is None:
                    then_node = child
                else:
                    else_node = child
        
        # Validate required nodes
        if not condition_node:
            # Fallback: treat as expression
            return self._process_expression(stmt_node, prev_id)
        
        if not then_node:
            # No then branch, treat as expression
            return self._process_expression(stmt_node, prev_id)
        
        # Create decision node
        decision_id = self._new_node_id()
        condition_text = self.source_code[condition_node.start_byte:condition_node.end_byte].strip()
        condition_text = condition_text.strip('()')
        
        decision_node = CFGNode(
            id=decision_id,
            type=NodeType.DECISION,
            text=condition_text,
            ast_node=condition_node,
            start_byte=condition_node.start_byte,
            end_byte=condition_node.end_byte
        )
        self.nodes[decision_id] = decision_node
        self.edges.append(CFGEdge(from_id=prev_id, to_id=decision_id))
        
        # Process then branch
        then_id = self._process_statement(then_node, decision_id)
        self.edges.append(CFGEdge(from_id=decision_id, to_id=then_id, label="YES"))
        
        # Process else branch
        if else_node:
            else_id = self._process_statement(else_node, decision_id)
            self.edges.append(CFGEdge(from_id=decision_id, to_id=else_id, label="NO"))
            # Merge point
            merge_id = self._new_node_id()
            merge_node = CFGNode(
                id=merge_id,
                type=NodeType.PROCESS,
                text="merge"
            )
            self.nodes[merge_id] = merge_node
            self.edges.append(CFGEdge(from_id=then_id, to_id=merge_id))
            self.edges.append(CFGEdge(from_id=else_id, to_id=merge_id))
            return merge_id
        else:
            # No else: then branch goes to merge, decision goes to next
            merge_id = self._new_node_id()
            merge_node = CFGNode(
                id=merge_id,
                type=NodeType.PROCESS,
                text="merge"
            )
            self.nodes[merge_id] = merge_node
            self.edges.append(CFGEdge(from_id=then_id, to_id=merge_id))
            self.edges.append(CFGEdge(from_id=decision_id, to_id=merge_id, label="NO"))
            return merge_id
    
    def _process_while(self, stmt_node, prev_id: str) -> str:
        """Process while loop."""
        condition_node = None
        body_node = None
        
        for child in stmt_node.children:
            if child.type == 'parenthesized_expression':
                condition_node = child
            elif child.type in ['compound_statement', 'expression_statement']:
                body_node = child
        
        # Validate required nodes
        if not condition_node or not body_node:
            # Fallback: treat as expression
            return self._process_expression(stmt_node, prev_id)
        
        # Create loop condition node
        loop_id = self._new_node_id()
        condition_text = self.source_code[condition_node.start_byte:condition_node.end_byte].strip()
        condition_text = condition_text.strip('()')
        
        loop_node = CFGNode(
            id=loop_id,
            type=NodeType.LOOP,
            text=condition_text,
            ast_node=condition_node,
            start_byte=condition_node.start_byte,
            end_byte=condition_node.end_byte
        )
        self.nodes[loop_id] = loop_node
        self.edges.append(CFGEdge(from_id=prev_id, to_id=loop_id))
        
        # Process body
        body_id = self._process_statement(body_node, loop_id)
        self.edges.append(CFGEdge(from_id=loop_id, to_id=body_id, label="YES"))
        
        # Body loops back to condition
        self.edges.append(CFGEdge(from_id=body_id, to_id=loop_id))
        
        # Exit loop
        exit_id = self._new_node_id()
        exit_node = CFGNode(
            id=exit_id,
            type=NodeType.PROCESS,
            text="exit_loop"
        )
        self.nodes[exit_id] = exit_node
        self.edges.append(CFGEdge(from_id=loop_id, to_id=exit_id, label="NO"))
        
        return exit_id
    
    def _process_for(self, stmt_node, prev_id: str) -> str:
        """Process for loop."""
        init_node = None
        condition_node = None
        update_node = None
        body_node = None
        
        for child in stmt_node.children:
            if child.type == 'for_range_loop':
                # C++11 range-based for
                return self._process_for_range(stmt_node, prev_id)
            elif child.type == 'for_statement':
                # Extract init, condition, update, body
                parts = child.children
                if len(parts) >= 4:
                    init_node = parts[0] if parts[0].type != '(' else None
                    condition_node = parts[1] if len(parts) > 1 else None
                    update_node = parts[2] if len(parts) > 2 else None
                    body_node = parts[3] if len(parts) > 3 else None
        
        # If not found in children, try direct children
        if not init_node:
            for child in stmt_node.children:
                if child.type in ['declaration', 'expression_statement'] and init_node is None:
                    init_node = child
                elif child.type == 'parenthesized_expression' and condition_node is None:
                    condition_node = child
                elif child.type in ['compound_statement', 'expression_statement'] and body_node is None:
                    body_node = child
        
        # Create init node
        init_id = prev_id
        if init_node:
            init_id = self._process_statement(init_node, prev_id)
        
        # Create loop condition node
        loop_id = self._new_node_id()
        condition_text = "true"
        if condition_node:
            condition_text = self.source_code[condition_node.start_byte:condition_node.end_byte].strip()
            condition_text = condition_text.strip('()')
        
        loop_node = CFGNode(
            id=loop_id,
            type=NodeType.LOOP,
            text=condition_text,
            ast_node=condition_node,
            start_byte=condition_node.start_byte if condition_node else 0,
            end_byte=condition_node.end_byte if condition_node else 0
        )
        self.nodes[loop_id] = loop_node
        self.edges.append(CFGEdge(from_id=init_id, to_id=loop_id))
        
        # Process body
        body_id = self._process_statement(body_node, loop_id)
        self.edges.append(CFGEdge(from_id=loop_id, to_id=body_id, label="YES"))
        
        # Process update
        update_id = body_id
        if update_node:
            update_id = self._process_statement(update_node, body_id)
            self.edges.append(CFGEdge(from_id=body_id, to_id=update_id))
        
        # Update loops back to condition
        self.edges.append(CFGEdge(from_id=update_id, to_id=loop_id))
        
        # Exit loop
        exit_id = self._new_node_id()
        exit_node = CFGNode(
            id=exit_id,
            type=NodeType.PROCESS,
            text="exit_loop"
        )
        self.nodes[exit_id] = exit_node
        self.edges.append(CFGEdge(from_id=loop_id, to_id=exit_id, label="NO"))
        
        return exit_id
    
    def _process_for_range(self, stmt_node, prev_id: str) -> str:
        """Process C++11 range-based for loop."""
        # Simplified: treat as while loop
        return self._process_while(stmt_node, prev_id)
    
    def _process_do_while(self, stmt_node, prev_id: str) -> str:
        """Process do-while loop."""
        body_node = None
        condition_node = None
        
        for child in stmt_node.children:
            if child.type in ['compound_statement', 'expression_statement']:
                body_node = child
            elif child.type == 'parenthesized_expression':
                condition_node = child
        
        # Validate required nodes
        if not body_node or not condition_node:
            # Fallback: treat as expression
            return self._process_expression(stmt_node, prev_id)
        
        # Process body first
        body_id = self._process_statement(body_node, prev_id)
        
        # Create loop condition node
        loop_id = self._new_node_id()
        condition_text = self.source_code[condition_node.start_byte:condition_node.end_byte].strip()
        condition_text = condition_text.strip('()')
        
        loop_node = CFGNode(
            id=loop_id,
            type=NodeType.LOOP,
            text=condition_text,
            ast_node=condition_node,
            start_byte=condition_node.start_byte,
            end_byte=condition_node.end_byte
        )
        self.nodes[loop_id] = loop_node
        self.edges.append(CFGEdge(from_id=body_id, to_id=loop_id))
        self.edges.append(CFGEdge(from_id=loop_id, to_id=body_id, label="YES"))
        
        # Exit loop
        exit_id = self._new_node_id()
        exit_node = CFGNode(
            id=exit_id,
            type=NodeType.PROCESS,
            text="exit_loop"
        )
        self.nodes[exit_id] = exit_node
        self.edges.append(CFGEdge(from_id=loop_id, to_id=exit_id, label="NO"))
        
        return exit_id
    
    def _process_switch(self, stmt_node, prev_id: str) -> str:
        """Process switch statement."""
        expression_node = None
        body_node = None
        
        for child in stmt_node.children:
            if child.type == 'parenthesized_expression':
                expression_node = child
            elif child.type == 'compound_statement':
                body_node = child
        
        # Validate required nodes
        if not expression_node or not body_node:
            # Fallback: treat as expression
            return self._process_expression(stmt_node, prev_id)
        
        # Create switch node
        switch_id = self._new_node_id()
        expr_text = self.source_code[expression_node.start_byte:expression_node.end_byte].strip()
        expr_text = expr_text.strip('()')
        
        switch_node = CFGNode(
            id=switch_id,
            type=NodeType.SWITCH,
            text=f"switch({expr_text})",
            ast_node=expression_node,
            start_byte=expression_node.start_byte,
            end_byte=expression_node.end_byte
        )
        self.nodes[switch_id] = switch_node
        self.edges.append(CFGEdge(from_id=prev_id, to_id=switch_id))
        
        # Process switch body
        return self._process_switch_body(body_node, switch_id)
    
    def _process_switch_body(self, body_node, switch_id: str) -> str:
        """Process switch statement body."""
        case_nodes = []
        default_node = None
        
        # Extract cases and default
        for child in body_node.children:
            if child.type == 'case_statement':
                case_nodes.append(child)
            elif child.type == 'default_statement':
                default_node = child
        
        # Process cases
        merge_id = self._new_node_id()
        merge_node = CFGNode(
            id=merge_id,
            type=NodeType.PROCESS,
            text="merge_switch"
        )
        self.nodes[merge_id] = merge_node
        
        for case_node in case_nodes:
            case_id = self._process_case(case_node, switch_id)
            self.edges.append(CFGEdge(from_id=case_id, to_id=merge_id))
        
        if default_node:
            default_id = self._process_default(default_node, switch_id)
            self.edges.append(CFGEdge(from_id=default_id, to_id=merge_id))
        
        return merge_id
    
    def _process_case(self, case_node, prev_id: str) -> str:
        """Process case label."""
        case_id = self._new_node_id()
        case_text = self.source_code[case_node.start_byte:case_node.end_byte].strip()
        
        case_cfg_node = CFGNode(
            id=case_id,
            type=NodeType.CASE,
            text=case_text,
            ast_node=case_node,
            start_byte=case_node.start_byte,
            end_byte=case_node.end_byte
        )
        self.nodes[case_id] = case_cfg_node
        self.edges.append(CFGEdge(from_id=prev_id, to_id=case_id))
        
        # Process case body
        for child in case_node.children:
            if child.type not in ['case', 'default']:
                return self._process_statement(child, case_id)
        
        return case_id
    
    def _process_default(self, default_node, prev_id: str) -> str:
        """Process default label."""
        default_id = self._new_node_id()
        default_text = "default"
        
        default_cfg_node = CFGNode(
            id=default_id,
            type=NodeType.DEFAULT,
            text=default_text,
            ast_node=default_node,
            start_byte=default_node.start_byte,
            end_byte=default_node.end_byte
        )
        self.nodes[default_id] = default_cfg_node
        self.edges.append(CFGEdge(from_id=prev_id, to_id=default_id))
        
        # Process default body
        for child in default_node.children:
            if child.type != 'default':
                return self._process_statement(child, default_id)
        
        return default_id
    
    def _process_break(self, stmt_node, prev_id: str) -> str:
        """Process break statement."""
        break_id = self._new_node_id()
        break_node = CFGNode(
            id=break_id,
            type=NodeType.BREAK,
            text="break",
            ast_node=stmt_node,
            start_byte=stmt_node.start_byte,
            end_byte=stmt_node.end_byte
        )
        self.nodes[break_id] = break_node
        self.edges.append(CFGEdge(from_id=prev_id, to_id=break_id))
        
        return break_id
    
    def _process_continue(self, stmt_node, prev_id: str) -> str:
        """Process continue statement."""
        continue_id = self._new_node_id()
        continue_node = CFGNode(
            id=continue_id,
            type=NodeType.CONTINUE,
            text="continue",
            ast_node=stmt_node,
            start_byte=stmt_node.start_byte,
            end_byte=stmt_node.end_byte
        )
        self.nodes[continue_id] = continue_node
        self.edges.append(CFGEdge(from_id=prev_id, to_id=continue_id))
        
        return continue_id
    
    def _process_expression(self, stmt_node, prev_id: str) -> str:
        """Process expression statement."""
        node_id = self._new_node_id()
        text = self.source_code[stmt_node.start_byte:stmt_node.end_byte].strip()
        if not text:
            text = "statement"
        
        node = CFGNode(
            id=node_id,
            type=NodeType.PROCESS,
            text=text,
            ast_node=stmt_node,
            start_byte=stmt_node.start_byte,
            end_byte=stmt_node.end_byte
        )
        self.nodes[node_id] = node
        self.edges.append(CFGEdge(from_id=prev_id, to_id=node_id))
        
        return node_id
    
    def _process_declaration(self, stmt_node, prev_id: str) -> str:
        """Process declaration statement."""
        return self._process_expression(stmt_node, prev_id)
