# Agent C++ Function Flowchart Compiler

A compiler-style static analysis tool that converts C++ functions into Mermaid flowcharts using a validated pseudo-code intermediate model and open-source LLM-powered generation.

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
LLM-based Mermaid Flowchart Generator (Open-Source)
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
- **Open-Source LLM**: Uses local models via Ollama or Hugging Face transformers (no API keys required)
- **GPU Support**: Optimized for 64GB GPU servers with CUDA support
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
6. **Mermaid Generation**: Uses open-source LLM to generate Mermaid flowchart from validated IR
7. **Mermaid Validation**: Validates Mermaid syntax and control flow
8. **Metrics Calculation**: Computes complexity metrics

### LLM Backends

The compiler supports two open-source backends:

#### 1. Ollama (Recommended - Default)

Ollama is the easiest way to run local LLMs. It handles model management and inference automatically.

**Advantages:**
- Easy setup and model management
- Automatic GPU utilization
- No complex dependencies
- Supports many models (Llama, Mistral, CodeLlama, etc.)

**Setup:**
```bash
# Install Ollama from https://ollama.ai
# Pull a model (recommended for code generation)
ollama pull llama3.2
# Or for larger models on 64GB GPU
ollama pull llama3.1:70b
```

#### 2. Hugging Face Transformers

Direct integration with Hugging Face models for maximum control.

**Advantages:**
- Direct access to Hugging Face model hub
- Full control over model loading and inference
- Supports quantization and optimization
- Best for custom models

**Setup:**
```bash
# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install transformers
pip install transformers accelerate
```

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

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for 64GB GPU server)
- Ollama (for Ollama backend) OR PyTorch with CUDA (for transformers backend)

### Step 1: Clone Repository

```bash
git clone https://github.com/vishal9359/Agent8.git
cd Agent8
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Setup LLM Backend

#### Option A: Ollama (Recommended)

```bash
# Install Ollama from https://ollama.ai
# For Linux/Mac:
curl -fsSL https://ollama.ai/install.sh | sh

# For Windows: Download from https://ollama.ai/download

# Pull a model (recommended models for code generation)
ollama pull llama3.2        # ~2GB, fast, good quality
ollama pull codellama       # Code-specific model
ollama pull llama3.1:70b    # Larger model for 64GB GPU
```

#### Option B: Hugging Face Transformers

```bash
# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install transformers
pip install transformers accelerate

# Optional: For 8-bit quantization (saves memory)
pip install bitsandbytes
```

## Usage

### Basic Usage (Ollama - Default)

```bash
# Make sure Ollama is running
ollama serve

# In another terminal, compile a function
python main.py example.cpp --function CreateVolume
```

### Using Different Models

```bash
# Use a different Ollama model
python main.py example.cpp --function CreateVolume --model codellama

# Use transformers backend
python main.py example.cpp --function CreateVolume \
    --backend transformers \
    --model meta-llama/Llama-2-7b-chat-hf
```

### With Sub-Function Expansion

```bash
python main.py example.cpp \
    --function CreateVolume \
    --sub-functions AllocateSpace,UpdateMetadata
```

### Custom Ollama URL

```bash
# If Ollama is running on a different host/port
python main.py example.cpp \
    --function CreateVolume \
    --ollama-url http://192.168.1.100:11434
```

### Command-Line Options

```
Options:
  -f, --function TEXT        Function name to compile (default: first function)
  -s, --sub-functions TEXT   Comma-separated list of sub-functions to expand
  -o, --output-dir TEXT      Output directory (default: output)
  -m, --model TEXT           Model name (default: llama3.2 for Ollama)
  -b, --backend [ollama|transformers]
                             Backend to use (default: ollama)
  --ollama-url TEXT          Ollama API base URL (default: http://localhost:11434)
  --device [cuda|cpu]        Device for transformers (default: cuda)
```

## Recommended Models for 64GB GPU

### Ollama Models

- **llama3.1:70b** - Best quality, requires ~40GB VRAM
- **llama3.2:3b** - Fast, good quality, ~2GB VRAM
- **codellama:34b** - Code-specific, ~20GB VRAM
- **mistral:7b** - Balanced quality/speed, ~4GB VRAM

### Hugging Face Models

- **meta-llama/Llama-2-70b-chat-hf** - High quality
- **codellama/CodeLlama-34b-Instruct-hf** - Code-specific
- **mistralai/Mistral-7B-Instruct-v0.2** - Fast and efficient

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
6. Generate Mermaid flowchart using local LLM
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
7. **No API Keys Required**: Fully open-source, runs entirely locally

## Error Handling

The compiler classifies errors:

- `IRValidationError`: IR validation failures
- `MermaidSyntaxError`: Mermaid syntax errors
- `MermaidParseError`: Mermaid parse errors
- `MermaidRenderError`: Mermaid render errors
- `LLMComplianceError`: LLM output compliance issues

All errors trigger retry mechanisms with the same prompt and IR.

## Troubleshooting

### Ollama Connection Error

```
Error: Cannot connect to Ollama at http://localhost:11434
```

**Solution:**
1. Make sure Ollama is running: `ollama serve`
2. Check if the model is pulled: `ollama list`
3. Verify the URL: `--ollama-url http://localhost:11434`

### CUDA Out of Memory (Transformers)

**Solution:**
1. Use a smaller model
2. Enable 8-bit quantization: Install `bitsandbytes` and use `--device cpu` temporarily
3. Use Ollama instead (handles memory better)

### Model Not Found

**Solution:**
- For Ollama: `ollama pull <model-name>`
- For Transformers: Check model name on Hugging Face Hub

## Sub-Function Expansion

Sub-function expansion allows inlining of specified functions:

- Only expands allowed sub-functions
- Maximum expansion depth = 1
- Detects and prevents expansion cycles
- All other function calls remain atomic nodes
- No caller tracing or project crawling

## Performance Tips for 64GB GPU

1. **Use Ollama**: Better memory management and GPU utilization
2. **Use Larger Models**: With 64GB GPU, use 70B models for better quality
3. **Batch Processing**: Process multiple functions in sequence
4. **Model Caching**: Ollama caches models automatically
5. **CUDA Optimization**: Ensure CUDA drivers are up to date

## Contributing

Contributions are welcome! Please ensure:

- Code follows the compiler-style architecture
- All validation gates are maintained
- Deterministic processing is preserved
- Tests are added for new features
- Documentation is updated

## License

MIT License

## Author

Agent8 - C++ Function Flowchart Compiler
