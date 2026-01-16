"""
LLM-based Mermaid Flowchart Generator.
Generates Mermaid code from validated PseudoCodeModel IR.
"""

from typing import Dict, Optional
import json
import os


class MermaidGenerator:
    """LLM-based Mermaid generator."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4", use_anthropic: bool = False):
        """
        Initialize Mermaid generator.
        
        Args:
            api_key: API key for LLM (OpenAI or Anthropic)
            model: Model name (default: gpt-4)
            use_anthropic: Use Anthropic Claude instead of OpenAI
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.use_anthropic = use_anthropic
        self.temperature = 0  # Deterministic generation
    
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
                mermaid_code = self._call_llm(prompt)
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
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API to generate Mermaid code."""
        if self.use_anthropic:
            return self._call_anthropic(prompt)
        else:
            return self._call_openai(prompt)
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a Mermaid flowchart generator. Return only valid Mermaid code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=2000
            )
            
            mermaid_code = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if mermaid_code.startswith("```"):
                lines = mermaid_code.split("\n")
                mermaid_code = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            
            return mermaid_code
        except ImportError:
            raise Exception("OpenAI library not installed. Install with: pip install openai")
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API."""
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=self.api_key)
            
            response = client.messages.create(
                model=self.model if self.model.startswith("claude") else "claude-3-sonnet-20240229",
                max_tokens=2000,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            mermaid_code = response.content[0].text.strip()
            
            # Remove markdown code blocks if present
            if mermaid_code.startswith("```"):
                lines = mermaid_code.split("\n")
                mermaid_code = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            
            return mermaid_code
        except ImportError:
            raise Exception("Anthropic library not installed. Install with: pip install anthropic")
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
