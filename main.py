"""
Agent C++ Function Flowchart Compiler
Main CLI entry point for the compiler pipeline.
Uses open-source local models (Ollama or Hugging Face transformers).
"""

import click
import json
import os
import sys
import re
from pathlib import Path
from typing import Optional, List

from ast_parser import ASTParser
from cfg_extractor import CFGExtractor
from cfg_canonicalizer import CFGCanonicalizer
from pseudo_code_model import PseudoCodeModel
from validator import Validator
from mermaid_generator import MermaidGenerator
from mermaid_repair import MermaidRepair
from complexity_metrics import ComplexityMetrics
from sub_function_expander import SubFunctionExpander


class CompilerPipeline:
    """Compiler-style pipeline for C++ function flowchart generation."""
    
    def __init__(self, model: str = "llama3.2", backend: str = "ollama",
                 ollama_base_url: str = "http://localhost:11434",
                 device: str = "cuda"):
        """
        Initialize compiler pipeline.
        
        Args:
            model: Model name (default: llama3.2 for Ollama)
            backend: Backend to use - "ollama" or "transformers" (default: ollama)
            ollama_base_url: Ollama API base URL (default: http://localhost:11434)
            device: Device for transformers ("cuda" or "cpu", default: cuda)
        """
        self.ast_parser = ASTParser()
        self.cfg_extractor = CFGExtractor()
        self.cfg_canonicalizer = CFGCanonicalizer()
        self.validator = Validator()
        self.mermaid_generator = MermaidGenerator(
            model=model,
            backend=backend,
            ollama_base_url=ollama_base_url,
            device=device
        )
        self.mermaid_repair = MermaidRepair()
        self.complexity_calculator = ComplexityMetrics()
        self.sub_function_expander = None
    
    def compile(self, file_path: str, function_name: Optional[str] = None,
                sub_functions: Optional[List[str]] = None,
                output_dir: str = "output") -> dict:
        """
        Compile C++ function to Mermaid flowchart.
        
        Args:
            file_path: Path to C++ source file
            function_name: Function name to compile (None for first function)
            sub_functions: List of sub-functions to expand
            output_dir: Output directory for results
            
        Returns:
            Dictionary with mermaid, ir, and metrics
        """
        # Step 1: Parse C++ file
        click.echo(f"Parsing C++ file: {file_path}")
        ast_data = self.ast_parser.parse_file(file_path)
        
        # Step 2: Find function
        function_node = self.ast_parser.find_function(ast_data, function_name)
        if not function_node:
            # List available functions for better error message
            available_functions = self.ast_parser.list_functions(ast_data)
            error_msg = f"Function '{function_name}' not found in {file_path}"
            if available_functions:
                error_msg += f"\n\nAvailable functions in this file:\n"
                for func in available_functions[:20]:  # Limit to first 20
                    error_msg += f"  - {func}\n"
                if len(available_functions) > 20:
                    error_msg += f"  ... and {len(available_functions) - 20} more\n"
            raise ValueError(error_msg)
        
        func_name = function_node["name"]
        click.echo(f"Found function: {func_name}")
        
        # Step 3: Extract function body
        function_body = self.ast_parser.get_function_body(function_node, ast_data["source"])
        if not function_body:
            raise ValueError(f"Function body not found for {func_name}")
        
        # Step 4: Extract raw CFG
        click.echo("Extracting Control Flow Graph...")
        nodes, edges = self.cfg_extractor.extract(function_body, ast_data["source"])
        
        # Step 5: Canonicalize CFG
        click.echo("Canonicalizing CFG...")
        canonical_nodes, canonical_edges = self.cfg_canonicalizer.canonicalize(nodes, edges)
        
        # Step 6: Generate PseudoCodeModel IR
        click.echo("Generating PseudoCodeModel IR...")
        ir_model = PseudoCodeModel.from_cfg(
            canonical_nodes, canonical_edges,
            func_name, file_path, f"Function: {func_name}"
        )
        
        # Step 7: Expand sub-functions if requested
        if sub_functions:
            click.echo(f"Expanding sub-functions: {sub_functions}")
            self.sub_function_expander = SubFunctionExpander(
                allowed_functions=sub_functions,
                max_depth=1
            )
            ir_model_dict = self.sub_function_expander.expand(
                ir_model.to_dict(), ast_data, func_name
            )
            ir_model = PseudoCodeModel(
                ir_model_dict["entry_function"],
                ir_model_dict["file"],
                ir_model_dict.get("description", "")
            )
            ir_model.steps = ir_model_dict["steps"]
            ir_model.edges = ir_model_dict["edges"]
        
        # Step 8: Validate IR
        click.echo("Validating PseudoCodeModel IR...")
        ir_dict = ir_model.to_dict()
        is_valid, errors = self.validator.validate_ir(ir_dict)
        
        if not is_valid:
            click.echo(f"IR validation failed: {errors}", err=True)
            raise ValueError(f"IR validation failed: {errors}")
        
        click.echo("IR validation passed")
        
        # Step 9: Generate Mermaid using LLM with validation feedback loop
        click.echo(f"Generating Mermaid flowchart using {self.mermaid_generator.backend} ({self.mermaid_generator.model})...")
        max_retries = 5  # Increased retries for better success rate
        mermaid_code = None
        validation_errors = None
        
        for attempt in range(1, max_retries + 1):
            try:
                # Generate with validation feedback from previous attempt
                click.echo(f"Attempt {attempt}/{max_retries}...")
                mermaid_code = self.mermaid_generator.generate(
                    ir_dict, 
                    validation_errors=validation_errors,
                    attempt=attempt,
                    max_retries=1
                )
                
                # Always try to repair common issues automatically
                click.echo("  Applying automatic repairs...")
                mermaid_code = self.mermaid_repair.repair(mermaid_code)
                
                # Validate Mermaid
                is_valid, errors = self.validator.validate_mermaid(mermaid_code)
                if is_valid:
                    click.echo("✓ Mermaid validation passed")
                    break
                else:
                    validation_errors = errors
                    click.echo(f"✗ Mermaid validation failed (attempt {attempt}/{max_retries})", err=True)
                    click.echo(f"  Errors: {', '.join(errors[:3])}{'...' if len(errors) > 3 else ''}", err=True)
                    
                    if attempt < max_retries:
                        click.echo(f"  Retrying with improved prompt and repairs...", err=True)
                    else:
                        # Last attempt - try one more repair pass
                        click.echo("  Applying final repair pass...")
                        mermaid_code = self.mermaid_repair.repair(mermaid_code)
                        is_valid, errors = self.validator.validate_mermaid(mermaid_code)
                        if is_valid:
                            click.echo("✓ Mermaid validation passed after final repair")
                            break
                        
                        # Still failed - show all errors
                        click.echo(f"\nAll validation errors:", err=True)
                        for i, error in enumerate(errors, 1):
                            click.echo(f"  {i}. {error}", err=True)
                        raise ValueError(f"Mermaid validation failed after {max_retries} attempts and repairs")
                        
            except ValueError:
                # Re-raise validation errors
                raise
            except Exception as e:
                click.echo(f"✗ Mermaid generation failed (attempt {attempt}/{max_retries}): {str(e)}", err=True)
                if attempt == max_retries:
                    raise
        
        if not mermaid_code:
            raise ValueError("Failed to generate valid Mermaid code")
        
        # Step 10: Calculate complexity metrics
        click.echo("Calculating complexity metrics...")
        metrics = self.complexity_calculator.calculate(
            canonical_nodes, canonical_edges, ir_dict
        )
        
        # Step 11: Write output
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = Path(output_dir) / f"{func_name}_flowchart.mmd"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(mermaid_code)
        click.echo(f"Mermaid written to: {output_file}")
        
        ir_file = Path(output_dir) / f"{func_name}_ir.json"
        with open(ir_file, 'w', encoding='utf-8') as f:
            f.write(ir_model.to_json(indent=2))
        click.echo(f"IR written to: {ir_file}")
        
        metrics_file = Path(output_dir) / f"{func_name}_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        click.echo(f"Metrics written to: {metrics_file}")
        
        return {
            "mermaid": mermaid_code,
            "ir": ir_dict,
            "metrics": metrics,
            "output_files": {
                "mermaid": str(output_file),
                "ir": str(ir_file),
                "metrics": str(metrics_file)
            }
        }


@click.command()
@click.argument('file_path', type=str)
@click.option('--function', '-f', help='Function name to compile (default: first function)')
@click.option('--sub-functions', '-s', help='Comma-separated list of sub-functions to expand')
@click.option('--output-dir', '-o', default='output', help='Output directory (default: output)')
@click.option('--model', '-m', default='llama3.2', help='Model name (default: llama3.2 for Ollama)')
@click.option('--backend', '-b', type=click.Choice(['ollama', 'transformers'], case_sensitive=False),
              default='ollama', help='Backend to use: ollama or transformers (default: ollama)')
@click.option('--ollama-url', default='http://localhost:11434',
              help='Ollama API base URL (default: http://localhost:11434)')
@click.option('--device', type=click.Choice(['cuda', 'cpu'], case_sensitive=False),
              default='cuda', help='Device for transformers: cuda or cpu (default: cuda)')
def main(file_path, function, sub_functions, output_dir, model, backend, ollama_url, device):
    """
    Agent C++ Function Flowchart Compiler
    
    Compiles C++ functions to Mermaid flowcharts using a compiler-style pipeline.
    Uses open-source local models (Ollama or Hugging Face transformers).
    
    Examples:
        # Using Ollama (default)
        python main.py example.cpp --function CreateVolume
        
        # Using Ollama with custom model
        python main.py example.cpp --function CreateVolume --model llama3.1
        
        # Using Hugging Face transformers
        python main.py example.cpp --function CreateVolume --backend transformers --model meta-llama/Llama-2-7b-chat-hf
        
        # With sub-function expansion
        python main.py example.cpp --function CreateVolume --sub-functions AllocateSpace,UpdateMetadata
    """
    try:
        # Handle Windows paths - the path might have backslashes that need proper handling
        import os
        
        # Debug: log the received path
        original_path = file_path
        
        # Remove quotes if present (user might have quoted the path)
        file_path = file_path.strip('"\'')
        
        # Normalize using os.path which handles Windows paths correctly
        normalized_path = os.path.normpath(file_path)
        
        # Try multiple path resolution strategies
        path_candidates = [
            normalized_path,
            os.path.abspath(normalized_path),
            file_path,  # Original (after quote removal)
            original_path,  # Very original
        ]
        
        # Try to resolve if path exists
        for candidate in path_candidates[:2]:  # Only try normalized and absolute
            try:
                if Path(candidate).exists():
                    resolved = Path(candidate).resolve()
                    if resolved.exists():
                        path_candidates.insert(0, str(resolved))
                        break
            except (OSError, ValueError):
                continue
        
        # Remove duplicates while preserving order
        seen = set()
        path_candidates = [p for p in path_candidates if p and p not in seen and not seen.add(p)]
        
        # Find the first path that exists
        resolved_path = None
        for candidate in path_candidates:
            try:
                path_obj = Path(candidate)
                if path_obj.exists() and path_obj.is_file():
                    resolved_path = str(path_obj.resolve())
                    break
            except (OSError, ValueError, RuntimeError) as e:
                continue
        
        if not resolved_path:
            # Last attempt: try to reconstruct if backslashes were stripped
            # Check if path looks mangled (has drive letter but no path separators)
            if ':' in normalized_path and (os.sep not in normalized_path and '/' not in normalized_path and '\\' not in normalized_path):
                # Try to reconstruct the path by inserting backslashes
                # Pattern: DriveLetter:folder1folder2folder3file.ext
                # We'll try common patterns
                reconstructed_paths = []
                
                # Try to reconstruct the path by inserting backslashes intelligently
                mangled = normalized_path
                
                if ':' in mangled:
                    parts = mangled.split(':', 1)
                    if len(parts) == 2:
                        drive = parts[0] + ':'
                        rest = parts[1]
                        
                        # Strategy 1: Insert backslashes before known folder patterns (in order)
                        # Handle: git-projectposeidonossrciofrontend_ioaio.cpp
                        # Result: git-project\poseidonos\src\io\frontend_io\aio.cpp
                        test_path = rest
                        
                        # Insert backslash before each known pattern (but not if already has one)
                        # Order matters - process longer patterns first to avoid partial matches
                        known_folders = ['frontend_io', 'backend_io', 'git-project', 'poseidonos', 'src', 'io']
                        for folder in known_folders:
                            # Insert backslash before the folder name if it appears (only once per occurrence)
                            # Use word boundary to avoid partial matches
                            pattern = r'(?<!\\)(?<!' + re.escape(folder[0]) + r')' + re.escape(folder)
                            test_path = re.sub(pattern, r'\\' + folder, test_path)
                        
                        # DON'T insert backslash before file extension - file extensions are part of filenames
                        # Instead, ensure backslash is before the filename (which ends with extension)
                        # The filename should already be separated by the folder matching above
                        
                        # Clean up: ensure single backslashes, add leading backslash after drive
                        test_path = re.sub(r'\\\\+', r'\\', test_path)
                        if not test_path.startswith('\\'):
                            test_path = '\\' + test_path
                        
                        reconstructed = drive + test_path
                        reconstructed_paths.append(reconstructed)
                        
                        # Strategy 2: Insert before hyphens and underscores (folder separators)
                        test_path2 = rest
                        # Insert before hyphen (but keep the hyphen) - this separates "git" and "project"
                        test_path2 = re.sub(r'(-)', r'\\\1', test_path2)
                        # Insert before underscore (but keep the underscore) - this separates "frontend" and "io"
                        test_path2 = re.sub(r'(_)', r'\\\1', test_path2)
                        # DON'T insert before file extension - it's part of the filename
                        # Clean up
                        test_path2 = re.sub(r'\\\\+', r'\\', test_path2)
                        if not test_path2.startswith('\\'):
                            test_path2 = '\\' + test_path2
                        
                        reconstructed2 = drive + test_path2
                        if reconstructed2 != reconstructed:
                            reconstructed_paths.append(reconstructed2)
                        
                        # Strategy 3: Smart word boundary detection
                        # Insert backslash before transitions: lowercase-to-lowercase word boundaries
                        # This handles cases like "poseidonossrc" -> "poseidonos\src"
                        test_path3 = rest
                        # Pattern: lowercase letter followed by lowercase letter that starts a known word
                        for folder in ['poseidonos', 'src', 'io', 'frontend_io', 'backend_io']:
                            # Find the folder name and insert backslash before it (if not already there)
                            pattern = r'(?<=[a-z])' + re.escape(folder)
                            test_path3 = re.sub(pattern, r'\\' + folder, test_path3, count=1)
                        # DON'T insert before file extension
                        # Clean up
                        test_path3 = re.sub(r'\\\\+', r'\\', test_path3)
                        if not test_path3.startswith('\\'):
                            test_path3 = '\\' + test_path3
                        
                        reconstructed3 = drive + test_path3
                        if reconstructed3 not in reconstructed_paths:
                            reconstructed_paths.append(reconstructed3)
                        
                        # Try each reconstructed path
                        for recon_path in reconstructed_paths:
                            try:
                                path_obj = Path(recon_path)
                                if path_obj.exists() and path_obj.is_file():
                                    resolved_path = str(path_obj.resolve())
                                    click.echo(f"✓ Successfully reconstructed path: {resolved_path}")
                                    break
                            except (OSError, ValueError, RuntimeError):
                                continue
                
                if not resolved_path:
                    click.echo(f"Error: Path appears to be malformed. Backslashes were stripped by the shell.", err=True)
                    click.echo(f"  Received: {repr(original_path)}", err=True)
                    click.echo(f"", err=True)
                    click.echo(f"  SOLUTION: Quote the path in your command:", err=True)
                    click.echo(f"  python main.py \"D:\\git-project\\poseidonos\\src\\io\\frontend_io\\aio.cpp\" --function _SendUserCompletion", err=True)
                    click.echo(f"", err=True)
                    click.echo(f"  Or use forward slashes (no quotes needed):", err=True)
                    click.echo(f"  python main.py D:/git-project/poseidonos/src/io/frontend_io/aio.cpp --function _SendUserCompletion", err=True)
                    sys.exit(1)
            else:
                click.echo(f"Error: File not found: {normalized_path}", err=True)
                click.echo(f"  Original path received: {repr(original_path)}", err=True)
                click.echo(f"  Normalized path: {normalized_path}", err=True)
                click.echo(f"  Current directory: {os.getcwd()}", err=True)
                click.echo(f"  Tried {len(path_candidates)} path variations", err=True)
            sys.exit(1)
        
        if not resolved_path:
            click.echo(f"Error: Could not resolve file path", err=True)
            sys.exit(1)
        
        file_path = resolved_path
        
        # Verify it's actually a file
        if not Path(file_path).is_file():
            click.echo(f"Error: Path is not a file: {file_path}", err=True)
            sys.exit(1)
        
        # Parse sub-functions
        sub_func_list = None
        if sub_functions:
            sub_func_list = [f.strip() for f in sub_functions.split(',')]
        
        # Remove parentheses from function name if present (e.g., _SendUserCompletion() -> _SendUserCompletion)
        if function:
            function = function.strip()
            # Remove trailing parentheses and any parameters
            function = re.sub(r'\([^)]*\)\s*$', '', function)
        
        # Initialize pipeline
        pipeline = CompilerPipeline(
            model=model,
            backend=backend,
            ollama_base_url=ollama_url,
            device=device
        )
        
        # Compile
        result = pipeline.compile(
            file_path=file_path,
            function_name=function,
            sub_functions=sub_func_list,
            output_dir=output_dir
        )
        
        click.echo("\n✓ Compilation successful!")
        click.echo(f"  Cyclomatic Complexity: {result['metrics']['cyclomatic_complexity']}")
        click.echo(f"  Nodes: {result['metrics']['node_count']}")
        click.echo(f"  Edges: {result['metrics']['edge_count']}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
