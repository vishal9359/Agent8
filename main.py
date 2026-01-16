"""
Agent C++ Function Flowchart Compiler
Main CLI entry point for the compiler pipeline.
"""

import click
import json
import os
import sys
from pathlib import Path
from typing import Optional, List

from ast_parser import ASTParser
from cfg_extractor import CFGExtractor
from cfg_canonicalizer import CFGCanonicalizer
from pseudo_code_model import PseudoCodeModel
from validator import Validator
from mermaid_generator import MermaidGenerator
from complexity_metrics import ComplexityMetrics
from sub_function_expander import SubFunctionExpander


class CompilerPipeline:
    """Compiler-style pipeline for C++ function flowchart generation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4", 
                 use_anthropic: bool = False):
        """
        Initialize compiler pipeline.
        
        Args:
            api_key: LLM API key
            model: LLM model name
            use_anthropic: Use Anthropic instead of OpenAI
        """
        self.ast_parser = ASTParser()
        self.cfg_extractor = CFGExtractor()
        self.cfg_canonicalizer = CFGCanonicalizer()
        self.validator = Validator()
        self.mermaid_generator = MermaidGenerator(api_key, model, use_anthropic)
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
            raise ValueError(f"Function '{function_name}' not found in {file_path}")
        
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
        
        # Step 9: Generate Mermaid using LLM
        click.echo("Generating Mermaid flowchart using LLM...")
        max_retries = 3
        mermaid_code = None
        
        for attempt in range(max_retries):
            try:
                mermaid_code = self.mermaid_generator.generate(ir_dict, max_retries=1)
                
                # Validate Mermaid
                is_valid, errors = self.validator.validate_mermaid(mermaid_code)
                if is_valid:
                    click.echo("Mermaid validation passed")
                    break
                else:
                    click.echo(f"Mermaid validation failed (attempt {attempt + 1}/{max_retries}): {errors}", err=True)
                    if attempt == max_retries - 1:
                        raise ValueError(f"Mermaid validation failed after {max_retries} attempts: {errors}")
            except Exception as e:
                click.echo(f"Mermaid generation failed (attempt {attempt + 1}/{max_retries}): {str(e)}", err=True)
                if attempt == max_retries - 1:
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
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--function', '-f', help='Function name to compile (default: first function)')
@click.option('--sub-functions', '-s', help='Comma-separated list of sub-functions to expand')
@click.option('--output-dir', '-o', default='output', help='Output directory (default: output)')
@click.option('--api-key', envvar='OPENAI_API_KEY', help='OpenAI API key (or set OPENAI_API_KEY env var)')
@click.option('--model', default='gpt-4', help='LLM model name (default: gpt-4)')
@click.option('--anthropic', is_flag=True, help='Use Anthropic Claude instead of OpenAI')
@click.option('--anthropic-key', envvar='ANTHROPIC_API_KEY', help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
def main(file_path, function, sub_functions, output_dir, api_key, model, anthropic, anthropic_key):
    """
    Agent C++ Function Flowchart Compiler
    
    Compiles C++ functions to Mermaid flowcharts using a compiler-style pipeline.
    
    Example:
        python main.py example.cpp --function CreateVolume --sub-functions AllocateSpace,UpdateMetadata
    """
    try:
        # Determine API key
        if anthropic:
            api_key = anthropic_key or api_key
            if not api_key:
                click.echo("Error: Anthropic API key required. Set ANTHROPIC_API_KEY env var or use --anthropic-key", err=True)
                sys.exit(1)
        else:
            if not api_key:
                click.echo("Error: OpenAI API key required. Set OPENAI_API_KEY env var or use --api-key", err=True)
                sys.exit(1)
        
        # Parse sub-functions
        sub_func_list = None
        if sub_functions:
            sub_func_list = [f.strip() for f in sub_functions.split(',')]
        
        # Initialize pipeline
        pipeline = CompilerPipeline(api_key, model, anthropic)
        
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
