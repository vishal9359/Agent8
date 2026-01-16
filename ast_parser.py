"""
AST Parser using Tree-sitter for C++ function extraction.
Deterministic AST extraction from C++ source code.
"""

from tree_sitter import Language, Parser
import os


class ASTParser:
    """Deterministic C++ AST parser using Tree-sitter."""
    
    def __init__(self):
        """Initialize Tree-sitter C++ parser."""
        try:
            import tree_sitter_cpp as tscpp
            self.language = Language(tscpp.language())
        except Exception:
            # Fallback: try to load from shared library
            try:
                self.language = Language('build/my-languages.so', 'cpp')
            except Exception:
                # Try to build language
                Language.build_library(
                    'build/my-languages.so',
                    ['tree-sitter-cpp']
                )
                self.language = Language('build/my-languages.so', 'cpp')
        
        self.parser = Parser(self.language)
    
    def parse_file(self, file_path: str) -> dict:
        """
        Parse C++ file and return AST.
        
        Args:
            file_path: Path to C++ source file
            
        Returns:
            Dictionary with 'tree' and 'source' keys
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        tree = self.parser.parse(bytes(source_code, 'utf8'))
        return {
            'tree': tree,
            'source': source_code,
            'file_path': file_path
        }
    
    def parse_string(self, source_code: str) -> dict:
        """
        Parse C++ source code string and return AST.
        
        Args:
            source_code: C++ source code string
            
        Returns:
            Dictionary with 'tree' and 'source' keys
        """
        tree = self.parser.parse(bytes(source_code, 'utf8'))
        return {
            'tree': tree,
            'source': source_code,
            'file_path': None
        }
    
    def find_function(self, ast_data: dict, function_name: str = None) -> dict:
        """
        Find function definition in AST.
        
        Args:
            ast_data: AST data from parse_file or parse_string
            function_name: Optional function name to find
            
        Returns:
            Function node and its source code
        """
        tree = ast_data['tree']
        source = ast_data['source']
        root_node = tree.root_node
        
        def traverse(node):
            """Traverse AST to find function definitions."""
            if node.type == 'function_definition':
                func_name_node = None
                for child in node.children:
                    if child.type == 'function_declarator':
                        for subchild in child.children:
                            if subchild.type == 'identifier':
                                func_name_node = subchild
                                break
                
                if func_name_node:
                    func_name = source[func_name_node.start_byte:func_name_node.end_byte]
                    if function_name is None or func_name == function_name:
                        func_source = source[node.start_byte:node.end_byte]
                        return {
                            'node': node,
                            'name': func_name,
                            'source': func_source,
                            'start_byte': node.start_byte,
                            'end_byte': node.end_byte
                        }
            
            for child in node.children:
                result = traverse(child)
                if result:
                    return result
            
            return None
        
        return traverse(root_node)
    
    def get_function_body(self, function_node: dict, source: str) -> dict:
        """
        Extract function body from function node.
        
        Args:
            function_node: Function node dictionary
            source: Source code string
            
        Returns:
            Function body node
        """
        node = function_node['node']
        
        # Find compound statement (function body)
        for child in node.children:
            if child.type == 'compound_statement':
                body_source = source[child.start_byte:child.end_byte]
                return {
                    'node': child,
                    'source': body_source,
                    'start_byte': child.start_byte,
                    'end_byte': child.end_byte
                }
        
        return None
