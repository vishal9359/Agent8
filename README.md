# Agent C++ Function Flowchart Compiler

A compiler-style static analysis tool that converts C++ functions into Mermaid flowcharts using a validated pseudo-code intermediate model and LLM-powered generation.

## Overview

Agent is a real static-analysis compiler that follows a deterministic pipeline:

```
C++ Function
    ↓
AST Parsing (Tree-sitter)
    ↓
Function Control-Flow Extraction
    ↓
Structured Pseudo Code Model (JSON IR)
    ↓
Validation Gate (Retry if invalid)
    ↓
LLM-based Mermaid Flowchart Generator
    ↓
Validation Gate (Retry if invalid)
    ↓
Mermaid Output + JSON IR + Complexity Metrics
```

## Features

- **Deterministic AST Extraction**: Uses Tree-sitter for reliable C++ parsing
- **CFG Canonicalization**: Normalizes control flow graphs before IR generation
- **PseudoCodeModel IR**: LLM-optimized JSON intermediate representation
- **Validation Gates**: Comprehensive validation at IR and Mermaid stages
- **LLM-Powered Generation**: Uses GPT-4 or Claude to generate Mermaid flowcharts
- **Complexity Metrics**: Calculates cyclomatic complexity and other metrics
- **Sub-Function Expansion**: Optionally expands allowed sub-functions

## Architecture

### Compiler Pipeline

1. **AST Parsing**: Tree-sitter parses C++ source code into an Abstract Syntax Tree
2. **CFG Extraction**: Extracts raw Control Flow Graph from AST
3. **CFG Canonicalization**: Normalizes CFG according to canonicalization rules:
   - Merge multiple return blocks into single End node
   - Normalize implicit returns into explicit return nodes
   - Normalize switch fallthrough into explicit edges
   - Normalize break/continue targets
   - Convert short-circuit boolean logic into explicit decision nodes
   - Remove unreachable blocks
   - Collapse empty blocks
   - Ensure all paths reach End unless infinite loop exists
4. **IR Generation**: Converts canonicalized CFG to PseudoCodeModel JSON IR
5. **IR Validation**: Validates IR structure, connectivity, and correctness
6. **Mermaid Generation**: Uses LLM to generate Mermaid flowchart from validated IR
7. **Mermaid Validation**: Validates Mermaid syntax and control flow
8. **Metrics Calculation**: Computes complexity metrics

### PseudoCodeModel IR

The Intermediate Representation (IR) is a JSON structure optimized for LLM understanding:

```json
{
  "entry_function": "CreateVolume",
  "file": "volume.cpp",
  "description": "Creates a storage volume after validation",
  "steps": [
    {
      "id": "start",
      "type": "start",
      "text": "Start"
    },
    {
      "id": "p1",
      "type": "process",
      "text": "Validate user request"
    },
    {
      "id": "d1",
      "type": "decision",
      "text": "Is request valid?"
    },
    {
      "id": "p2",
      "type": "process",
      "text": "Allocate storage space"
    },
    {
      "id": "end",
      "type": "end",
      "text": "End"
    }
  ],
  "edges": [
    { "from": "start", "to": "p1" },
    { "from": "p1", "to": "d1" },
    { "from": "d1", "to": "p2", "label": "YES" },
    { "from": "d1", "to": "end", "label": "NO" },
    { "from": "p2", "to": "end" }
  ]
}
```

#### Step Types

- `start`: Entry point
- `process`: Normal operation
- `decision`: Branch condition
- `loop`: Loop condition
- `switch`: Switch expression
- `case`: Case label
- `default`: Default label
- `break`: Break statement
- `continue`: Continue statement
- `return`: Return statement
- `throw`: Exception
- `end`: Exit

### Validation Gates

#### IR Validation

- JSON schema validation
- Graph connectivity validation
- Loop correctness validation
- Branch completeness
- Switch correctness
- Return correctness
- Break/continue correctness
- Structural validation (exactly one Start/End)
- Reachability validation

#### Mermaid Validation

- Syntax validation
- Control flow validation
- Structural validation
- Semantic validation
- Parse error detection

### Complexity Metrics

The compiler calculates:

- **Cyclomatic Complexity**: `E - N + 2P` where E=edges, N=nodes, P=connected components
- **Node Count**: Total number of nodes
- **Edge Count**: Total number of edges
- **Decision Count**: Number of decision nodes
- **Loop Count**: Number of loop nodes
- **Exception Paths**: Number of exception paths
- **Max Depth**: Longest path from entry to exit

## Installation

```bash
# Clone the repository
git clone https://github.com/vishal9359/Agent8.git
cd Agent8

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python main.py example.cpp --function CreateVolume
```

### With Sub-Function Expansion

```bash
python main.py example.cpp --function CreateVolume --sub-functions AllocateSpace,UpdateMetadata
```

### Using Anthropic Claude

```bash
python main.py example.cpp --function CreateVolume --anthropic --anthropic-key YOUR_KEY
```

### Command-Line Options

```
Options:
  -f, --function TEXT        Function name to compile (default: first function)
  -s, --sub-functions TEXT   Comma-separated list of sub-functions to expand
  -o, --output-dir TEXT      Output directory (default: output)
  --api-key TEXT             OpenAI API key (or set OPENAI_API_KEY env var)
  --model TEXT               LLM model name (default: gpt-4)
  --anthropic                Use Anthropic Claude instead of OpenAI
  --anthropic-key TEXT       Anthropic API key (or set ANTHROPIC_API_KEY env var)
```

## Example

Given a C++ function:

```cpp
int CreateVolume(const Request& req) {
    if (!validate(req)) {
        return -1;
    }
    
    int space = AllocateSpace(req.size);
    if (space < 0) {
        return -2;
    }
    
    UpdateMetadata(req.id, space);
    return 0;
}
```

The compiler will:

1. Parse the function
2. Extract CFG
3. Canonicalize CFG
4. Generate IR
5. Validate IR
6. Generate Mermaid flowchart
7. Validate Mermaid
8. Calculate metrics
9. Write outputs to `output/` directory

Output files:
- `CreateVolume_flowchart.mmd`: Mermaid flowchart
- `CreateVolume_ir.json`: PseudoCodeModel IR
- `CreateVolume_metrics.json`: Complexity metrics

## Output Format

### Mermaid Flowchart

```mermaid
flowchart TD
    S1([Start])
    D1{validate(req)}
    P1[AllocateSpace]
    P2[UpdateMetadata]
    R1([return -1])
    R2([return -2])
    R3([return 0])
    E1([End])
    
    S1 --> D1
    D1 -->|No| R1
    D1 -->|Yes| P1
    R1 --> E1
    P1 --> D2{space < 0}
    D2 -->|Yes| R2
    D2 -->|No| P2
    R2 --> E1
    P2 --> R3
    R3 --> E1
```

### Metrics JSON

```json
{
  "cyclomatic_complexity": 3,
  "node_count": 8,
  "edge_count": 9,
  "decision_count": 2,
  "loop_count": 0,
  "exception_paths": 0,
  "max_depth": 5
}
```

## Design Principles

1. **Deterministic Processing**: All stages are deterministic (LLM temperature = 0)
2. **PseudoCodeModel as Source of Truth**: IR is the authoritative representation
3. **Validation Gates**: Strict validation before proceeding to next stage
4. **Retry Mechanism**: Automatic retry on validation failures
5. **LLM-Optimized IR**: JSON structure designed for LLM understanding
6. **Compiler-Style Architecture**: Clear pipeline with intermediate representations

## Error Handling

The compiler classifies errors:

- `IRValidationError`: IR validation failures
- `MermaidSyntaxError`: Mermaid syntax errors
- `MermaidParseError`: Mermaid parse errors
- `MermaidRenderError`: Mermaid render errors
- `LLMComplianceError`: LLM output compliance issues

All errors trigger retry mechanisms with the same prompt and IR.

## Sub-Function Expansion

Sub-function expansion allows inlining of specified functions:

- Only expands allowed sub-functions
- Maximum expansion depth = 1
- Detects and prevents expansion cycles
- All other function calls remain atomic nodes
- No caller tracing or project crawling

## Contributing

Contributions are welcome! Please ensure:

- Code follows the compiler-style architecture
- All validation gates are maintained
- Deterministic processing is preserved
- Tests are added for new features

## License

MIT License

## Author

Agent8 - C++ Function Flowchart Compiler
