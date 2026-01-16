"""
LLM-based Mermaid Flowchart Generator.
Generates Mermaid code from validated PseudoCodeModel IR using open-source local models.
Supports Ollama and Hugging Face transformers.
"""

from typing import Dict, Optional
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
    
    def generate(self, ir_model: Dict, max_retries: int = 3) -> str:
        """
        Generate Mermaid code from PseudoCodeModel IR.
        
        Args:
            ir_model: PseudoCodeModel dictionary
            max_retries: Maximum retry attempts
            
        Returns:
            Mermaid code string
            
        Raises:
            Exception: If generation fails after max_retries
        """
        prompt = self._build_prompt(ir_model)
        
        for attempt in range(max_retries):
            try:
                if self.backend == "ollama":
                    mermaid_code = self._call_ollama(prompt)
                elif self.backend == "transformers":
                    mermaid_code = self._call_transformers(prompt)
                else:
                    raise ValueError(f"Unknown backend: {self.backend}")
                
                return mermaid_code
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"LLM Mermaid generation failed after {max_retries} attempts: {str(e)}")
                continue
        
        raise Exception(f"LLM Mermaid generation failed after {max_retries} attempts")
    
    def _build_prompt(self, ir_model: Dict) -> str:
        """Build LLM prompt for Mermaid generation."""
        ir_json = json.dumps(ir_model, indent=2)
        
        prompt = """You are a software architect.

Convert the following pseudo code model into a valid Mermaid flowchart.

Rules:
Graph Type

Always use:
flowchart TD

Node Types

Start node : ([Start])
End node : ([End])
Process : [statement]
Function call : [/func()/]
Decision : {condition}
Return : ([return value])
Break : ([break])
Continue : ([continue])
Switch decision : {switch(expr)}
Case label : [case X]

Edge Rules

All decision branches must be labeled:
Yes / No
True / False
case X
default

If-Else Rule

if (cond) { A } else { B }

D1{cond}
D1 -->|Yes| P1[A]
D1 -->|No| P2[B]

While Loop Rule

while(cond) { A }

D1{cond}
D1 -->|Yes| P1[A]
P1 --> D1
D1 -->|No| NEXT

For Loop Rule

for(init; cond; inc) { A }

P1[init]
D1{cond}
D1 -->|Yes| P2[A]
P2 --> P3[inc]
P3 --> D1
D1 -->|No| NEXT

Do-While Rule

do { A } while(cond)

P1[A]
D1{cond}
P1 --> D1
D1 -->|Yes| P1
D1 -->|No| NEXT

Switch Rule

switch(x) {
case 1: A; break;
case 2: B; break;
default: C;
}

D1{switch(x)}
D1 -->|case 1| P1[A]
P1 --> B1([break])
B1 --> NEXT

D1 -->|case 2| P2[B]
P2 --> B2([break])
B2 --> NEXT

D1 -->|default| P3[C]
P3 --> NEXT

Return Rule

return x;

R1([return x])
R1 --> END

Naming Convention

S1 = Start
E1 = End
D1 = Decision
P1 = Process
R1 = Return
B1 = Break
C1 = Continue

Structural Rules

Exactly one Start

Exactly one End

No orphan nodes

No missing branches

No dangling edges

Nested blocks must reconnect correctly

Break exits only loop or switch

Continue jumps to loop condition

Return exits function

Pseudo Code Model:
""" + ir_json + """

Return only Mermaid code. Do not include any explanation, reasoning, or markdown code blocks. Return only the raw Mermaid flowchart code starting with 'flowchart TD'."""
        
        return prompt
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API to generate Mermaid code."""
        try:
            # Prepare request
            url = f"{self.ollama_base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": 2000
                }
            }
            
            response = requests.post(url, json=payload, timeout=300)
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
                f"Cannot connect to Ollama at {self.ollama_base_url}. "
                "Make sure Ollama is running. Install from https://ollama.ai"
            )
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
                max_new_tokens=2000,
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
