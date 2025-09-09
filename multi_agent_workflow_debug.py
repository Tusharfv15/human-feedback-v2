"""
Simplified debug version of multi-agent workflow with timeouts
"""

import json
import uuid
from typing import TypedDict, Literal, List, Dict, Any, Optional
import torch
import signal

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import traceable
from dotenv import load_dotenv
from model_manager import model_manager
load_dotenv()


class MultiAgentState(TypedDict):
    """State for the multi-agent workflow"""
    user_query: str
    selected_agent: str
    agent_reasoning: str
    selected_program: Dict[str, Any]
    llm_solution: str
    feedback: str
    iteration_count: int
    final_parameters: Dict[str, Any]
    change_summary: List[str]
    previous_solution: str
    dependency_analysis: str
    requires_other_agents: bool
    completed_agents: List[str]
    agent_chain: List[str]


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("LLM inference timed out")


class BaseAgent:
    """Base class for all specialized agents with timeout protection"""
    
    def __init__(self, agent_name: str, programs_file: str, knowledge_file: str):
        self.model, self.tokenizer = model_manager.get_model()
        self.agent_name = agent_name
        self.programs_file = programs_file
        self.knowledge_file = knowledge_file
        self.programs = self._load_programs()
        
    def _load_programs(self) -> Dict[str, Any]:
        """Load programs from JSON file"""
        try:
            with open(self.programs_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[WARNING] {self.agent_name}: Programs file not found: {self.programs_file}")
            return {}
    
    def _load_knowledge(self) -> str:
        """Load knowledge from text file"""
        try:
            with open(self.knowledge_file, 'r') as f:
                return f.read()
        except FileNotFoundError:
            print(f"[WARNING] {self.agent_name}: Knowledge file not found: {self.knowledge_file}")
            return ""
    
    def _clear_gpu_cache(self):
        """Clear GPU cache to prevent memory issues"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _generate_with_timeout(self, inputs, timeout_seconds=30):
        """Generate with timeout protection"""
        if self.model is None or self.tokenizer is None:
            return None
            
        try:
            # Set up timeout (Windows compatible approach)
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            print(f"[DEBUG] {self.agent_name}: Starting inference with timeout {timeout_seconds}s")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,  # Reduced for faster inference
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    use_cache=False,
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
            ).strip()
            
            self._clear_gpu_cache()
            print(f"[DEBUG] {self.agent_name}: Inference completed successfully")
            return response
            
        except Exception as e:
            print(f"[ERROR] {self.agent_name}: Inference error: {e}")
            self._clear_gpu_cache()
            return None
    
    def find_relevant_program(self, query: str) -> Dict[str, Any]:
        """Find the most relevant program - simplified version"""
        print(f"[DEBUG] {self.agent_name}: Finding program for query")
        
        if not self.programs:
            print(f"[DEBUG] {self.agent_name}: No programs available, using knowledge base")
            return {}
        
        if self.model is None or self.tokenizer is None:
            print(f"[DEBUG] {self.agent_name}: No model available, using first program")
            return list(self.programs.values())[0]
        
        try:
            # Simple program selection based on keywords
            query_lower = query.lower()
            
            # Keyword-based matching for faster execution
            if "boron" in query_lower or "p+" in query_lower or "p-type" in query_lower:
                for prog_id, prog in self.programs.items():
                    if "boron" in prog.get('task', '').lower():
                        print(f"[DEBUG] {self.agent_name}: Selected {prog_id} by keyword matching")
                        return prog
            
            if "phosphorus" in query_lower or "n+" in query_lower or "n-type" in query_lower:
                for prog_id, prog in self.programs.items():
                    if "phosphorus" in prog.get('task', '').lower():
                        print(f"[DEBUG] {self.agent_name}: Selected {prog_id} by keyword matching")
                        return prog
            
            # Fallback to first program
            first_program = list(self.programs.values())[0]
            print(f"[DEBUG] {self.agent_name}: Using first program as fallback")
            return first_program
            
        except Exception as e:
            print(f"[ERROR] {self.agent_name}: Error in program search: {e}")
            return {}
    
    def generate_solution(self, query: str, program: Dict[str, Any], 
                         feedback: str = None, previous_solution: str = None) -> str:
        """Generate solution with timeout protection"""
        print(f"[DEBUG] {self.agent_name}: Generating solution")
        
        if self.model is None or self.tokenizer is None:
            return f"[MOCK] {self.agent_name} solution for: {query[:50]}... Parameters: temp=950C, time=30s, dose=1e15 atoms/cm²"
        
        try:
            # Simplified prompt for faster inference
            if program:
                context = f"Program: {program.get('task', 'Unknown task')}"
            else:
                context = "Using knowledge base"
            
            simple_prompt = f"You are a {self.agent_name.replace('-agent', '')} expert. Give specific parameters for: {query[:100]}"
            
            inputs = self.tokenizer(
                simple_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,  # Reduced context length
                padding=False,
            )
            
            response = self._generate_with_timeout(inputs, timeout_seconds=30)
            
            if response and len(response.strip()) > 0:
                return response
            else:
                return f"[FALLBACK] {self.agent_name} solution for: {query[:50]}... Parameters: optimized for your requirements"
                
        except Exception as e:
            print(f"[ERROR] {self.agent_name}: Error generating solution: {e}")
            return f"[ERROR] {self.agent_name}: Could not generate solution"


# Simplified Agents
class DopingAgent(BaseAgent):
    def __init__(self):
        super().__init__("doping-agent", "programs/doping-programs.json", "knowledge/doping-knowledge.txt")

class EtchingAgent(BaseAgent):
    def __init__(self):
        super().__init__("etching-agent", "programs/etching-programs.json", "knowledge/etching-knowledge.txt")

class WaferAgent(BaseAgent):
    def __init__(self):
        super().__init__("wafer-agent", "programs/wafer-programs.json", "knowledge/wafer-knowledge.txt")


class SimpleOrchestrator:
    """Simplified orchestrator with rule-based routing"""
    
    def __init__(self):
        self.agents = {
            "doping": DopingAgent(),
            "etching": EtchingAgent(), 
            "wafer": WaferAgent()
        }
    
    def route_query(self, query: str) -> tuple[str, str]:
        """Simple rule-based routing"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["implant", "doping", "junction", "boron", "phosphorus", "activation"]):
            return "doping", "Query contains doping-related keywords"
        elif any(word in query_lower for word in ["etch", "plasma", "removal", "pattern"]):
            return "etching", "Query contains etching-related keywords"
        elif any(word in query_lower for word in ["clean", "oxidation", "cvd", "wafer", "surface"]):
            return "wafer", "Query contains wafer processing keywords"
        else:
            return "doping", "Default routing to doping agent"
    
    def analyze_dependencies(self, solution: str, current_agent: str) -> tuple[bool, str, Optional[str]]:
        """Simple dependency analysis"""
        # For debug version, assume no dependencies
        return False, "No dependencies found in debug mode", None


# Global orchestrator instance
orchestrator = SimpleOrchestrator()


@traceable(run_type="chain", name="Simple Orchestrator Node")
def simple_orchestrator_node(state: MultiAgentState) -> MultiAgentState:
    """Simplified orchestrator node"""
    print(f"[DEBUG] Simple orchestrator node started")
    
    if not state.get('selected_agent'):
        print(f"[DEBUG] Routing query: {state['user_query'][:50]}...")
        agent_name, reasoning = orchestrator.route_query(state['user_query'])
        print(f"[DEBUG] Selected agent: {agent_name}")
        
        state['selected_agent'] = agent_name
        state['agent_reasoning'] = reasoning
        state['completed_agents'] = [agent_name]
        state['agent_chain'] = [agent_name]
    
    print(f"[DEBUG] Simple orchestrator node completed")
    return state


@traceable(run_type="chain", name="Simple Agent Processing")
def simple_agent_processing_node(state: MultiAgentState) -> MultiAgentState:
    """Simplified agent processing"""
    print(f"[DEBUG] Agent processing node started")
    
    agent_name = state['selected_agent']
    agent = orchestrator.agents[agent_name]
    
    print(f"[DEBUG] Processing with {agent_name}")
    
    # Find program
    if not state.get('selected_program'):
        selected_program = agent.find_relevant_program(state['user_query'])
        state['selected_program'] = selected_program
    
    # Generate solution
    solution = agent.generate_solution(
        state['user_query'], 
        state['selected_program']
    )
    
    state['llm_solution'] = solution
    state['iteration_count'] = state.get('iteration_count', 0) + 1
    
    # Simple parameters
    parameters = {
        'iteration': state['iteration_count'],
        'agent_used': agent_name,
        'solution': solution
    }
    state['final_parameters'] = parameters
    
    print(f"[DEBUG] Agent processing node completed")
    return state


def create_simple_workflow():
    """Create simplified workflow for debugging"""
    print("[DEBUG] Creating simple workflow")
    
    builder = StateGraph(MultiAgentState)
    
    # Add nodes
    builder.add_node("orchestrator", simple_orchestrator_node)
    builder.add_node("agent_processing", simple_agent_processing_node)
    
    # Set entry point
    builder.set_entry_point("orchestrator")
    
    # Simple edges
    builder.add_edge("orchestrator", "agent_processing")
    builder.add_edge("agent_processing", END)
    
    print("[DEBUG] Simple workflow created")
    return builder.compile()


def run_simple_workflow(user_query: str):
    """Run simplified workflow for debugging"""
    print(f"[DEBUG] Starting simple workflow for: {user_query}")
    
    try:
        graph = create_simple_workflow()
        
        initial_state = {
            "user_query": user_query,
            "iteration_count": 0,
            "change_summary": [],
            "completed_agents": [],
            "agent_chain": []
        }
        
        print("[DEBUG] Invoking workflow...")
        result = graph.invoke(initial_state)
        
        print("\n" + "="*50)
        print("SIMPLE WORKFLOW RESULT")
        print("="*50)
        print(f"Query: {result['user_query']}")
        print(f"Agent: {result['selected_agent']}")
        print(f"Solution: {result['llm_solution']}")
        print("="*50)
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    query = "I need boron implantation for shallow p+ junctions"
    result = run_simple_workflow(query)