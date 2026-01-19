"""
LLM-based Mermaid Flowchart Generator.
Generates Mermaid code from validated PseudoCodeModel IR using open-source local models.
Supports Ollama and Hugging Face transformers.
"""

from typing import Dict, Optional, List, Tuple
import json
import os
import requests


class MermaidGenerator:
    """LLM-based Mermaid generator using open-source local models."""
    
    def __init__(self, model: str = "llama3.2", backend: str = "ollama", 
                 ollama_base_url: str = "http://localhost:11434",
                 device: str = "cuda"):
        """
        Initialize Mermaid generator.
        
        Args:
            model: Model name (default: llama3.2 for Ollama, or HuggingFace model path)
            backend: Backend to use - "ollama" or "transformers" (default: ollama)
            ollama_base_url: Ollama API base URL (default: http://localhost:11434)
            device: Device for transformers ("cuda" or "cpu", default: cuda)
        """
        self.model = model
        self.backend = backend.lower()
        self.ollama_base_url = ollama_base_url
        self.device = device
        self.temperature = 0  # Deterministic generation
        
        # Initialize transformers if needed
        if self.backend == "transformers":
            self._init_transformers()
    
    def _init_transformers(self):
        """Initialize transformers backend."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            
            self.torch = torch
            self.AutoTokenizer = AutoTokenizer
            self.AutoModelForCausalLM = AutoModelForCausalLM
            self.pipeline = pipeline
            
            # Check if CUDA is available
            if self.device == "cuda" and not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU")
                self.device = "cpu"
            
            # Initialize pipeline as None, will be loaded on first use
            self._pipeline = None
        except ImportError:
            raise ImportError(
                "transformers library not installed. Install with: "
                "pip install torch transformers accelerate"
            )
    
    def generate(self, ir_model: Dict, validation_errors: Optional[List[str]] = None, 
                 attempt: int = 1, max_retries: int = 3) -> str:
        """
        Generate Mermaid code from PseudoCodeModel IR.
        
        Args:
            ir_model: PseudoCodeModel dictionary
            validation_errors: List of validation errors from previous attempt (if any)
            attempt: Current attempt number (1-based)
            max_retries: Maximum retry attempts
            
        Returns:
            Mermaid code string
            
        Raises:
            Exception: If generation fails after max_retries
        """
        prompt = self._build_prompt(ir_model, validation_errors, attempt)
        
        try:
            if self.backend == "ollama":
                mermaid_code = self._call_ollama(prompt)
            elif self.backend == "transformers":
                mermaid_code = self._call_transformers(prompt)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
            
            return mermaid_code
        except Exception as e:
            raise Exception(f"LLM Mermaid generation failed: {str(e)}")
    
    def _build_prompt(self, ir_model: Dict, validation_errors: Optional[List[str]] = None,
                     attempt: int = 1) -> str:
        """Build LLM prompt for Mermaid generation with validation feedback."""
        ir_json = json.dumps(ir_model, indent=2)
        
        # Base prompt
        prompt = """You are a software architect specializing in Mermaid flowcharts.

Convert the following pseudo code model into a valid Mermaid flowchart.

CRITICAL RULES - READ CAREFULLY:

1. Graph Type: Always start with 'flowchart TD'

2. Node Types and Syntax:
   - Start: S1([Start])
   - End: E1([End])
   - Process: P1[statement] - ONE simple statement per node, no control flow
   - Decision: D1{condition} - condition only, no statements
   - Return: R1([return value])
   - Break: B1([break])
   - Continue: C1([continue])
   
   CRITICAL TEXT RULES:
   - Text inside nodes MUST be simple and clean
   - DO NOT put multiple statements in one process node
   - DO NOT put control flow (if/while/for) inside process nodes
   - DO NOT put semicolons or complex expressions in node text
   - If text contains quotes, use single quotes or escape them
   - Keep node text short and readable (max 50 characters recommended)
   - Example CORRECT: P1[status = POS_IO_STATUS_SUCCESS]
   - Example WRONG: P1[int status == POS_IO_STATUS_SUCCESS; if (unlikely(_GetErrorCount() > 0)) {status = POS_IO_STATUS_FAIL;}]
   - For complex logic, split into multiple process nodes or use decision nodes

3. DECISION NODES - MOST IMPORTANT:
   EVERY decision node MUST have EXACTLY TWO outgoing edges:
   - One labeled |Yes| or |True| (for true condition)
   - One labeled |No| or |False| (for false condition)
   
   Example:
   D1{condition}
   D1 -->|Yes| P1[action if true]
   D1 -->|No| P2[action if false]
   
   NEVER create a decision node with only one branch!

4. Edge Labeling Rules:
   - Decision branches: |Yes| and |No| (or |True| and |False|)
   - Switch cases: |case 1|, |case 2|, |default|
   - All decision edges MUST be labeled

5. If-Else Pattern:
   D1{condition}
   D1 -->|Yes| P1[then branch]
   D1 -->|No| P2[else branch]
   P1 --> NEXT
   P2 --> NEXT

6. While Loop Pattern:
   D1{condition}
   D1 -->|Yes| P1[loop body]
   P1 --> D1
   D1 -->|No| NEXT

7. For Loop Pattern:
   P1[init]
   D1{condition}
   D1 -->|Yes| P2[body]
   P2 --> P3[increment]
   P3 --> D1
   D1 -->|No| NEXT

8. Switch Pattern:
   D1{switch(x)}
   D1 -->|case 1| P1[action 1]
   D1 -->|case 2| P2[action 2]
   D1 -->|default| P3[default action]
   P1 --> NEXT
   P2 --> NEXT
   P3 --> NEXT

9. Return Pattern:
   R1([return value])
   R1 --> E1([End])

10. Structural Requirements:
    - Exactly ONE Start node: S1([Start]) - define it ONCE
    - Exactly ONE End node: E1([End]) - define it ONCE
    - All decision nodes have TWO branches (Yes and No)
    - No orphan nodes
    - All paths eventually reach End
    - NO duplicate node definitions (each node ID defined only once)
    - NO undefined node references (all nodes in edges must be defined)
    - NO "NEXT" placeholder - use actual node IDs or End node

11. Naming Convention:
    - S1 = Start node (only one)
    - E1 = End node (only one)
    - D1, D2, D3 = Decision nodes
    - P1, P2, P3 = Process nodes
    - R1, R2 = Return nodes
    - B1, B2 = Break nodes
    - C1, C2 = Continue nodes

12. CRITICAL - Avoid These Errors:
    - DO NOT define the same node twice (e.g., S1([Start]) and S1[Start])
    - DO NOT reference undefined nodes in edges
    - DO NOT use "NEXT" as a node - replace with actual node ID or E1
    - DO NOT add Yes/No labels to Start or End node edges
    - DO NOT create edges to nodes that don't exist
    - Every node referenced in an edge MUST be defined first
    - DO NOT put multiple statements in one process node - split them
    - DO NOT put control flow (if/while/for) inside process nodes - use decision/loop nodes
    - DO NOT put unescaped quotes or special characters in node text
    - DO NOT put semicolons, braces, or complex expressions in node text
    - Keep process node text simple: one assignment, one function call, or one simple operation

"""
        
        # Add validation error feedback if this is a retry
        if validation_errors and attempt > 1:
            prompt += f"""
IMPORTANT: Previous attempt failed validation. Fix these errors:

VALIDATION ERRORS FROM PREVIOUS ATTEMPT:
"""
            for i, error in enumerate(validation_errors, 1):
                prompt += f"{i}. {error}\n"
            
            prompt += """
CRITICAL FIXES NEEDED:
- Ensure EVERY decision node has BOTH |Yes| and |No| branches
- Check that all decision edges are properly labeled
- Verify all nodes are connected correctly
- Make sure there is exactly one Start and one End node

"""
        
        # Add IR model
        prompt += f"""
Pseudo Code Model:
{ir_json}

INSTRUCTIONS:
1. Analyze the pseudo code model carefully
2. For each step in the model:
   - If type is "process": Create a simple process node with ONE statement only
     * Extract the main action from the text
     * Remove control flow statements (if/while/for)
     * Remove semicolons and complex expressions
     * Keep text under 50 characters if possible
     * Example: "int status = POS_IO_STATUS_SUCCESS" -> P1[status = POS_IO_STATUS_SUCCESS]
   - If type is "decision": Create a decision node with the condition only
     * Extract just the condition, not the full if statement
     * Example: "if (unlikely(_GetErrorCount() > 0))" -> D1{_GetErrorCount() > 0}
   - If type is "loop": Create a loop decision node with condition only
3. For EACH decision node, create TWO edges: one with |Yes| label, one with |No| label
4. Ensure all decision nodes have exactly 2 outgoing edges
5. Define each node ONLY ONCE (no duplicates)
6. Ensure all nodes referenced in edges are defined
7. Replace any "NEXT" placeholders with actual node IDs or E1
8. Do NOT add Yes/No labels to Start or End node edges
9. Escape special characters in node text (use single quotes if needed)
10. Split complex process steps into multiple simple process nodes
11. Generate valid Mermaid code starting with 'flowchart TD'
12. Return ONLY the Mermaid code, no explanations, no markdown blocks

VALIDATION CHECKLIST BEFORE RETURNING:
- [ ] Exactly one S1([Start]) defined
- [ ] Exactly one E1([End]) defined
- [ ] All decision nodes have |Yes| and |No| branches
- [ ] No duplicate node definitions
- [ ] All nodes in edges are defined
- [ ] No "NEXT" or undefined node references
- [ ] Start/End nodes don't have labeled edges

Generate the Mermaid flowchart now:"""
        
        return prompt
    
    def _check_ollama_health(self) -> Tuple[bool, str, List[str]]:
        """
        Check Ollama health and return status, version info, and available models.
        
        Returns:
            (is_healthy, version_info, available_models)
        """
        # Try to connect to Ollama
        try:
            # Check /api/version endpoint (most reliable)
            version_url = f"{self.ollama_base_url}/api/version"
            version_response = requests.get(version_url, timeout=5)
            
            if version_response.status_code == 200:
                version_info = version_response.json().get("version", "unknown")
                
                # Get available models
                models = []
                try:
                    tags_url = f"{self.ollama_base_url}/api/tags"
                    tags_response = requests.get(tags_url, timeout=5)
                    if tags_response.status_code == 200:
                        models_data = tags_response.json().get("models", [])
                        models = [m.get("name", "") for m in models_data]
                except:
                    pass
                
                return True, version_info, models
            else:
                return False, f"HTTP {version_response.status_code}", []
                
        except requests.exceptions.ConnectionError:
            return False, "Connection refused", []
        except requests.exceptions.Timeout:
            return False, "Connection timeout", []
        except Exception as e:
            return False, str(e), []
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API to generate Mermaid code."""
        try:
            # First, check Ollama health and get diagnostics
            is_healthy, version_info, available_models = self._check_ollama_health()
            
            if not is_healthy:
                error_msg = f"Cannot connect to Ollama at {self.ollama_base_url}.\n"
                error_msg += f"Connection status: {version_info}\n\n"
                error_msg += "Please ensure:\n"
                error_msg += "  1. Ollama is installed (https://ollama.ai)\n"
                error_msg += "  2. Ollama is running: 'ollama serve' or start Ollama service\n"
                error_msg += f"  3. Ollama is accessible at {self.ollama_base_url}\n"
                error_msg += "  4. Check firewall/network settings if using remote Ollama"
                raise Exception(error_msg)
            
            # Check if model exists (handle version tags like :latest, :7b, etc.)
            model_found = False
            if available_models:
                # Check exact match first
                if self.model in available_models:
                    model_found = True
                else:
                    # Check if model name matches without tag (e.g., llama3.2 vs llama3:latest)
                    model_base = self.model.split(':')[0] if ':' in self.model else self.model
                    for available_model in available_models:
                        available_base = available_model.split(':')[0] if ':' in available_model else available_model
                        if model_base == available_base or available_model.startswith(model_base + ':'):
                            # Use the available model instead
                            self.model = available_model
                            model_found = True
                            break
            
            if not model_found and available_models:
                error_msg = f"Model '{self.model}' not found in Ollama.\n\n"
                error_msg += f"Available models: {', '.join(available_models) if available_models else 'None'}\n\n"
                error_msg += f"Pull the model with: ollama pull {self.model}\n"
                error_msg += f"Or use an available model: --model {available_models[0]}"
                raise Exception(error_msg)
            
            # Try /api/chat endpoint first (newer Ollama versions, >= 0.1.0)
            # But if it fails, immediately fall back to /api/generate
            chat_url = f"{self.ollama_base_url}/api/chat"
            chat_payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a Mermaid flowchart generator. Return only valid Mermaid code."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": 3000
                }
            }
            
            chat_success = False
            try:
                response = requests.post(chat_url, json=chat_payload, timeout=300)
                if response.status_code == 200:
                    result = response.json()
                    mermaid_code = result.get("message", {}).get("content", "").strip()
                    if mermaid_code:
                        # Remove markdown code blocks if present
                        if mermaid_code.startswith("```"):
                            lines = mermaid_code.split("\n")
                            mermaid_code = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
                        return mermaid_code
                    chat_success = True
                elif response.status_code == 404:
                    # /api/chat not available, will try /api/generate
                    pass
                else:
                    response.raise_for_status()
            except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                # If /api/chat fails, fall back to /api/generate
                pass
            
            # Fallback to /api/generate endpoint (older Ollama versions or if /api/chat fails)
            generate_url = f"{self.ollama_base_url}/api/generate"
            generate_payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": 3000
                }
            }
            
            try:
                response = requests.post(generate_url, json=generate_payload, timeout=300)
                if response.status_code == 200:
                    result = response.json()
                    mermaid_code = result.get("response", "").strip()
                    
                    # Remove markdown code blocks if present
                    if mermaid_code.startswith("```"):
                        lines = mermaid_code.split("\n")
                        mermaid_code = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
                    
                    return mermaid_code
                else:
                    response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    # Both endpoints returned 404 - this is unusual
                    raise Exception(
                        f"Ollama API endpoints not found (404).\n"
                        f"Ollama version: {version_info}\n"
                        f"Tried endpoints: /api/chat, /api/generate\n\n"
                        "This might mean:\n"
                        "  1. Ollama version is incompatible - try updating: 'ollama update'\n"
                        "  2. Ollama service is not fully started - wait a few seconds and retry\n"
                        f"  3. Wrong base URL - current: {self.ollama_base_url}\n"
                        f"  4. Model '{self.model}' doesn't exist - pull it with: ollama pull {self.model}\n"
                        f"     Available models: {', '.join(available_models) if available_models else 'None'}"
                    )
                raise
            
        except requests.exceptions.ConnectionError as e:
            is_healthy, version_info, available_models = self._check_ollama_health()
            error_msg = f"Cannot connect to Ollama at {self.ollama_base_url}.\n"
            if version_info:
                error_msg += f"Connection status: {version_info}\n"
            error_msg += "\nPlease ensure:\n"
            error_msg += "  1. Ollama is installed (https://ollama.ai)\n"
            error_msg += "  2. Ollama is running: 'ollama serve' or start Ollama service\n"
            error_msg += f"  3. Ollama is accessible at {self.ollama_base_url}\n"
            error_msg += "  4. Check firewall/network settings if using remote Ollama"
            raise Exception(error_msg)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # Get fresh diagnostics
                is_healthy, version_info, available_models = self._check_ollama_health()
                raise Exception(
                    f"Ollama API endpoint not found (404).\n"
                    f"Ollama version: {version_info if is_healthy else 'unknown'}\n"
                    f"Tried endpoints: /api/chat, /api/generate\n\n"
                    "This might mean:\n"
                    "  1. Ollama version is incompatible - try updating: 'ollama update'\n"
                    "  2. Ollama service is not fully started - wait a few seconds and retry\n"
                    f"  3. Wrong base URL - current: {self.ollama_base_url}\n"
                    f"  4. Model '{self.model}' doesn't exist - pull it with: ollama pull {self.model}\n"
                    f"     Available models: {', '.join(available_models) if available_models else 'None'}"
                )
            else:
                raise Exception(f"Ollama API error ({e.response.status_code}): {str(e)}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API error: {str(e)}")
    
    def _call_transformers(self, prompt: str) -> str:
        """Call transformers (Hugging Face) to generate Mermaid code."""
        try:
            # Initialize pipeline if not already done
            if self._pipeline is None:
                print(f"Loading model {self.model}... This may take a while on first run.")
                device_map = "auto" if self.device == "cuda" else None
                torch_dtype = self.torch.float16 if self.device == "cuda" else self.torch.float32
                
                self._pipeline = self.pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.model,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    model_kwargs={"low_cpu_mem_usage": True}
                )
            
            # Generate
            system_prompt = "You are a Mermaid flowchart generator. Return only valid Mermaid code."
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
            outputs = self._pipeline(
                full_prompt,
                max_new_tokens=3000,  # Increased for better quality
                temperature=self.temperature,
                do_sample=(self.temperature > 0),
                pad_token_id=self._pipeline.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            mermaid_code = outputs[0]["generated_text"].strip()
            
            # Remove markdown code blocks if present
            if mermaid_code.startswith("```"):
                lines = mermaid_code.split("\n")
                mermaid_code = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            
            return mermaid_code
        except Exception as e:
            raise Exception(f"Transformers generation error: {str(e)}")
