"""
LLM-based Mermaid Flowchart Generator.
Generates Mermaid code from validated PseudoCodeModel IR using open-source local models.
Supports Ollama and Hugging Face transformers.
"""

from typing import Dict, Optional, List
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

2. Node Types:
   - Start: S1([Start])
   - End: E1([End])
   - Process: P1[statement]
   - Decision: D1{condition}
   - Return: R1([return value])
   - Break: B1([break])
   - Continue: C1([continue])

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
    - Exactly ONE Start node: S1([Start])
    - Exactly ONE End node: E1([End])
    - All decision nodes have TWO branches (Yes and No)
    - No orphan nodes
    - All paths eventually reach End

11. Naming Convention:
    - S1, S2 = Start nodes
    - E1, E2 = End nodes
    - D1, D2, D3 = Decision nodes
    - P1, P2, P3 = Process nodes
    - R1, R2 = Return nodes
    - B1, B2 = Break nodes
    - C1, C2 = Continue nodes

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
2. Identify all decision nodes (type: "decision")
3. For EACH decision node, create TWO edges: one with |Yes| label, one with |No| label
4. Ensure all decision nodes have exactly 2 outgoing edges
5. Generate valid Mermaid code starting with 'flowchart TD'
6. Return ONLY the Mermaid code, no explanations, no markdown blocks

Generate the Mermaid flowchart now:"""
        
        return prompt
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API to generate Mermaid code."""
        try:
            # First, check if Ollama is running
            try:
                health_url = f"{self.ollama_base_url}/api/tags"
                health_response = requests.get(health_url, timeout=5)
                if health_response.status_code == 404:
                    # Try alternative health check
                    health_url = f"{self.ollama_base_url}/api/version"
                    health_response = requests.get(health_url, timeout=5)
            except requests.exceptions.ConnectionError:
                raise Exception(
                    f"Cannot connect to Ollama at {self.ollama_base_url}.\n"
                    "Please ensure:\n"
                    "  1. Ollama is installed (https://ollama.ai)\n"
                    "  2. Ollama is running: 'ollama serve' or start Ollama service\n"
                    "  3. The model is pulled: 'ollama pull llama3.2'\n"
                    f"  4. Ollama is accessible at {self.ollama_base_url}"
                )
            
            # Try /api/chat endpoint first (newer Ollama versions)
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
            except requests.exceptions.RequestException:
                pass  # Fall back to /api/generate
            
            # Fallback to /api/generate endpoint (older Ollama versions)
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
            
            response = requests.post(generate_url, json=generate_payload, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            mermaid_code = result.get("response", "").strip()
            
            # Remove markdown code blocks if present
            if mermaid_code.startswith("```"):
                lines = mermaid_code.split("\n")
                mermaid_code = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            
            return mermaid_code
            
        except requests.exceptions.ConnectionError:
            raise Exception(
                f"Cannot connect to Ollama at {self.ollama_base_url}.\n"
                "Please ensure:\n"
                "  1. Ollama is installed (https://ollama.ai)\n"
                "  2. Ollama is running: 'ollama serve' or start Ollama service\n"
                "  3. The model is pulled: 'ollama pull llama3.2'\n"
                f"  4. Ollama is accessible at {self.ollama_base_url}"
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # Check if model exists
                try:
                    tags_url = f"{self.ollama_base_url}/api/tags"
                    tags_response = requests.get(tags_url, timeout=5)
                    if tags_response.status_code == 200:
                        models = tags_response.json().get("models", [])
                        model_names = [m.get("name", "") for m in models]
                        raise Exception(
                            f"Model '{self.model}' not found in Ollama.\n"
                            f"Available models: {', '.join(model_names) if model_names else 'None'}\n"
                            f"Pull the model with: ollama pull {self.model}"
                        )
                except:
                    pass
                
                raise Exception(
                    f"Ollama API endpoint not found (404).\n"
                    "This might mean:\n"
                    "  1. Ollama is not running - start with 'ollama serve'\n"
                    "  2. Wrong Ollama URL - check with: --ollama-url <url>\n"
                    "  3. Ollama version mismatch - try updating Ollama\n"
                    f"  4. Model '{self.model}' doesn't exist - pull it with: ollama pull {self.model}"
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
