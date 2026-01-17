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
        
        Supports both simple names and fully qualified names:
        - Simple: "_SendUserCompletion"
        - Fully qualified: "AioCompletion::_SendUserCompletion"
        
        Args:
            ast_data: AST data from parse_file or parse_string
            function_name: Optional function name to find (can be Class::Method or just Method)
            
        Returns:
            Function node and its source code
        """
        tree = ast_data['tree']
        source = ast_data['source']
        root_node = tree.root_node
        
        # Parse fully qualified name if provided
        target_class = None
        target_method = None
        if function_name and '::' in function_name:
            parts = function_name.split('::', 1)
            target_class = parts[0].strip()
            target_method = parts[1].strip()
        else:
            target_method = function_name
        
        def extract_function_name(node, source_str, class_context=None):
            """Extract function name from function_definition node."""
            # Look for function_declarator
            func_declarator = None
            for child in node.children:
                if child.type == 'function_declarator':
                    func_declarator = child
                    break
            
            if not func_declarator:
                return None, None
            
            # Extract identifier - could be field_identifier for member functions
            func_name = None
            qualified_class = None
            
            # Check for qualified_identifier first (for out-of-class definitions like Class::Method)
            for child in func_declarator.children:
                if child.type == 'qualified_identifier':
                    # Handle qualified identifiers like AioCompletion::_SendUserCompletion
                    # Structure: qualified_identifier -> [type_identifier, ::, field_identifier]
                    children_list = list(child.children)
                    # Find class name (type_identifier) and function name (field_identifier)
                    for i, subchild in enumerate(children_list):
                        if subchild.type == 'type_identifier':
                            qualified_class = source_str[subchild.start_byte:subchild.end_byte]
                        elif subchild.type == 'field_identifier':
                            func_name = source_str[subchild.start_byte:subchild.end_byte]
                        elif subchild.type == 'identifier' and not func_name:
                            # Fallback: might be just an identifier in qualified context
                            func_name = source_str[subchild.start_byte:subchild.end_byte]
                    break
                elif child.type == 'field_identifier':
                    func_name = source_str[child.start_byte:child.end_byte]
                    break
                elif child.type == 'identifier':
                    func_name = source_str[child.start_byte:child.end_byte]
                    break
            
            # Use qualified class if found, otherwise use context
            final_class = qualified_class if qualified_class else class_context
            
            return func_name, final_class
        
        def traverse(node, class_context=None, require_class_match=True):
            """Traverse AST to find function definitions.
            
            Args:
                node: AST node to traverse
                class_context: Current class name context (tracked during traversal)
                require_class_match: If True, require class name match for fully qualified names
            """
            # Update class context if we enter a class definition
            current_class = class_context
            if node.type == 'class_specifier':
                # Extract class name
                for child in node.children:
                    if child.type == 'type_identifier':
                        current_class = source[child.start_byte:child.end_byte]
                        break
            
            if node.type == 'function_definition':
                func_name, class_name = extract_function_name(node, source, current_class)
                
                if func_name:
                    # Use tracked class context if available
                    if not class_name:
                        class_name = current_class
                    
                    # Check if this matches the target
                    if function_name is None:
                        # No target specified, return first function
                        full_name = f"{class_name}::{func_name}" if class_name else func_name
                        func_source = source[node.start_byte:node.end_byte]
                        return {
                            'node': node,
                            'name': full_name,
                            'source': func_source,
                            'start_byte': node.start_byte,
                            'end_byte': node.end_byte
                        }
                    elif target_class and target_method:
                        # Match fully qualified name
                        class_matches = (not require_class_match) or (class_name == target_class)
                        if class_matches and func_name == target_method:
                            full_name = f"{class_name}::{func_name}" if class_name else func_name
                            func_source = source[node.start_byte:node.end_byte]
                            return {
                                'node': node,
                                'name': full_name,
                                'source': func_source,
                                'start_byte': node.start_byte,
                                'end_byte': node.end_byte
                            }
                    elif target_method:
                        # Match just method name (search anywhere)
                        if func_name == target_method:
                            full_name = f"{class_name}::{func_name}" if class_name else func_name
                            func_source = source[node.start_byte:node.end_byte]
                            return {
                                'node': node,
                                'name': full_name,
                                'source': func_source,
                                'start_byte': node.start_byte,
                                'end_byte': node.end_byte
                            }
            
            for child in node.children:
                result = traverse(child, current_class, require_class_match)
                if result:
                    return result
            
            return None
        
        # First try with exact class match
        result = traverse(root_node, class_context=None, require_class_match=True)
        
        # If not found with fully qualified name, try matching just the method name as fallback
        if not result and target_class and target_method:
            result = traverse(root_node, class_context=None, require_class_match=False)
        
        return result
    
    def list_functions(self, ast_data: dict) -> list:
        """
        List all functions in the AST.
        
        Args:
            ast_data: AST data from parse_file or parse_string
            
        Returns:
            List of function names (with class prefix if applicable)
        """
        tree = ast_data['tree']
        source = ast_data['source']
        root_node = tree.root_node
        functions = []
        
        def extract_function_name(node, source_str, class_context=None):
            """Extract function name from function_definition node."""
            func_declarator = None
            for child in node.children:
                if child.type == 'function_declarator':
                    func_declarator = child
                    break
            
            if not func_declarator:
                return None, None
            
            func_name = None
            qualified_class = None
            
            for child in func_declarator.children:
                if child.type == 'qualified_identifier':
                    children_list = list(child.children)
                    for subchild in children_list:
                        if subchild.type == 'type_identifier':
                            qualified_class = source_str[subchild.start_byte:subchild.end_byte]
                        elif subchild.type == 'field_identifier':
                            func_name = source_str[subchild.start_byte:subchild.end_byte]
                        elif subchild.type == 'identifier' and not func_name:
                            func_name = source_str[subchild.start_byte:subchild.end_byte]
                    break
                elif child.type == 'field_identifier':
                    func_name = source_str[child.start_byte:child.end_byte]
                    break
                elif child.type == 'identifier':
                    func_name = source_str[child.start_byte:child.end_byte]
                    break
            
            final_class = qualified_class if qualified_class else class_context
            return func_name, final_class
        
        def traverse(node, class_context=None):
            """Traverse AST to find all function definitions."""
            current_class = class_context
            if node.type == 'class_specifier':
                for child in node.children:
                    if child.type == 'type_identifier':
                        current_class = source[child.start_byte:child.end_byte]
                        break
            
            if node.type == 'function_definition':
                func_name, class_name = extract_function_name(node, source, current_class)
                if func_name:
                    full_name = f"{class_name}::{func_name}" if class_name else func_name
                    functions.append(full_name)
            
            for child in node.children:
                traverse(child, current_class)
        
        traverse(root_node)
        return functions
    
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
