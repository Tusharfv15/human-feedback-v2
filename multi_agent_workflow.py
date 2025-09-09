"""
Multi-Agent Semiconductor Processing Workflow with Human Feedback Loop
"""

import json
import uuid
from typing import TypedDict, Literal, List, Dict, Any, Optional
import torch

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


class BaseAgent:
    """Base class for all specialized agents"""

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
            print(
                f"[WARNING] {self.agent_name}: Programs file not found: {self.programs_file}")
            return {}

    def _load_knowledge(self) -> str:
        """Load knowledge from text file"""
        try:
            with open(self.knowledge_file, 'r') as f:
                return f.read()
        except FileNotFoundError:
            print(
                f"[WARNING] {self.agent_name}: Knowledge file not found: {self.knowledge_file}")
            return ""

    def create_alpaca_prompt(self, instruction: str, system_behavior: str = None) -> str:
        """Create Alpaca format prompt for the finetuned model"""
        if system_behavior:
            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{system_behavior}

### Response:
"""
        else:
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
        return prompt

    def _clear_gpu_cache(self):
        """Clear GPU cache to prevent memory issues"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate fallback response when LLM fails"""
        return f"[FALLBACK] {self.agent_name}: Unable to process: {prompt[:100]}..."

    @traceable(run_type="tool")
    def find_relevant_program(self, query: str) -> Dict[str, Any]:
        """Find the most relevant program using the finetuned LLM"""
        if not self.programs or self.model is None or self.tokenizer is None:
            return {}

        try:
            program_list = ""
            for program_id, program_data in self.programs.items():
                program_list += f"\n{program_id}: {program_data['task']}\n"
                for sub_htp in program_data.get('sub_htps', []):
                    program_list += f"  - {sub_htp['task']}\n"

            instruction = f"You are a specialized {self.agent_name.replace('-agent', '')} expert in semiconductor processing. Analyze the query and determine if any of the available programs could be helpful for this task.\n\nIf you find a relevant program that could be useful (even if not a perfect match), respond with ONLY the program_id (e.g., 'program_1', 'program_2', etc.).\nIf NO program is relevant or could be adapted for this query, respond with EXACTLY: 'NONE'\n\nExamples:\nQuery: I need boron implantation for shallow junctions\nResponse: program_2\n\nQuery: I need to measure wafer thickness\nResponse: NONE\n\nSelect the most suitable program if available, but use 'NONE' only when programs are clearly not applicable.\n\nAvailable Programs:{program_list}"

            alpaca_prompt = self.create_alpaca_prompt(instruction, query)

            inputs = self.tokenizer(
                alpaca_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1536,
                padding=False,
            )

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if hasattr(self.model, 'past_key_values'):
                self.model.past_key_values = None

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]                           :], skip_special_tokens=True
            ).strip()

            self._clear_gpu_cache()

            # Extract program_id from response
            if response.upper() == "NONE":
                # No relevant program found, return empty dict to trigger knowledge base usage
                print(
                    f"[INFO] {self.agent_name}: No relevant program found, will use knowledge base")
                return {}
            elif response in self.programs:
                return self.programs[response]
            else:
                # Try to find program_id in response text
                for program_id in self.programs.keys():
                    if program_id in response.lower():
                        return self.programs[program_id]
                # If still no match found, treat as NONE
                print(
                    f"[INFO] {self.agent_name}: Could not parse program selection '{response}', will use knowledge base")
                return {}

        except Exception as e:
            print(f"[ERROR] {self.agent_name}: Error in program search: {e}")
            print(
                f"[INFO] {self.agent_name}: Falling back to knowledge base due to error")
            self._clear_gpu_cache()
            return {}

    @traceable(run_type="llm")
    def generate_solution(self, query: str, program: Dict[str, Any],
                          feedback: str = None, previous_solution: str = None) -> str:
        """Generate solution using the finetuned LLM"""
        if self.model is None or self.tokenizer is None:
            return f"[MOCK RESPONSE from {self.agent_name}] Processing: {query[:100]}..."

        try:
            if not program:
                print(f"[DEBUG] No program, loading from knowledge base")
                # Use knowledge base when no program is found
                knowledge = self._load_knowledge()
                program_details = f"Using knowledge base for {self.agent_name}:\n\n{knowledge[:1000]}..."
            else:
                program_details = f"Program: {program['task']}\n\nAvailable parameters and ranges:\n"
                for sub_htp in program.get('sub_htps', []):
                    program_details += f"\nTask: {sub_htp['task']}\n"
                    for param in sub_htp.get('table', []):
                        program_details += f"- {param['parameter']}: {param['typical_range']} {param['units']}\n"
                    if 'note' in sub_htp:
                        program_details += f"Note: {sub_htp['note']}\n"

            instruction = f"You are a specialized {self.agent_name.replace('-agent', '')} expert in semiconductor processing. Based on the program data or knowledge provided, give specific parameter recommendations with explanations.\n\n{program_details}\n\nProvide specific parameter values with explanations:"

            user_input = f"User Query: {query}"

            # ADD DEBUGGING HERE
            print(f"[DEBUG] generate_solution called:")
            print(f"[DEBUG] Agent: {self.agent_name}")
            print(f"[DEBUG] Has feedback: {bool(feedback)}")
            print(f"[DEBUG] Has previous_solution: {bool(previous_solution)}")

            if previous_solution and feedback:
                user_input += f"\n\nPrevious Solution:\n{previous_solution}"
                user_input += f"\n\nUser Feedback: {feedback}\nAnalyze the feedback and implement the specific adjustments requested. Modify the previous solution parameters accordingly and explain how each change addresses the feedback."
                print(f"[DEBUG] Using previous solution + feedback mode")
                print(f"[DEBUG] Feedback: '{feedback}'")
            elif feedback:
                user_input += f"\n\nUser Feedback: {feedback}\nImplement the requested parameter adjustments. Provide updated recommendations with specific values and explain how these changes will resolve the issues mentioned in the feedback."
                print(f"[DEBUG] Using feedback-only mode")
                print(f"[DEBUG] Feedback: '{feedback}'")
            else:
                print(f"[DEBUG] Using initial generation mode (no feedback)")

            # Print the full prompt being sent to LLM
            alpaca_prompt = self.create_alpaca_prompt(instruction, user_input)
            print(f"[DEBUG] Full prompt length: {len(alpaca_prompt)} chars")
            # Show last 500 chars
            print(f"[DEBUG] Prompt preview: {alpaca_prompt[-500:]}")

            inputs = self.tokenizer(
                alpaca_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1536,
                padding=False,
            )

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if hasattr(self.model, 'past_key_values'):
                self.model.past_key_values = None

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    top_p=0.9,
                    use_cache=False,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]                           :], skip_special_tokens=True
            ).strip()

            self._clear_gpu_cache()

            # ADD DEBUGGING FOR RESPONSE
            print(f"[DEBUG] LLM raw response length: {len(response)} chars")
            print(f"[DEBUG] LLM raw response preview: {response[:300]}...")

            if not response or len(response.strip()) < 1:
                return self._generate_fallback_response(query)

            return response

        except Exception as e:
            print(f"[ERROR] {self.agent_name}: Error generating response: {e}")
            self._clear_gpu_cache()
            return self._generate_fallback_response(query)


# Specialized Agents
class DopingAgent(BaseAgent):
    def __init__(self):
        super().__init__("doping-agent", "programs/doping-programs.json",
                         "knowledge/doping-knowledge.txt")


class EtchingAgent(BaseAgent):
    def __init__(self):
        super().__init__("etching-agent", "programs/etching-programs.json",
                         "knowledge/etching-knowledge.txt")


class WaferAgent(BaseAgent):
    def __init__(self):
        super().__init__("wafer-agent", "programs/wafer-programs.json",
                         "knowledge/wafer-knowledge.txt")


class Orchestrator:
    """Orchestrator that routes queries to appropriate agents"""

    def __init__(self):
        self.model, self.tokenizer = model_manager.get_model()
        self.agents = {
            "doping": DopingAgent(),
            "etching": EtchingAgent(),
            "wafer": WaferAgent()
        }

    def create_alpaca_prompt(self, instruction: str, system_behavior: str = None) -> str:
        """Create Alpaca format prompt for the finetuned model"""
        if system_behavior:
            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{system_behavior}

### Response:
"""
        else:
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
        return prompt

    def _clear_gpu_cache(self):
        """Clear GPU cache to prevent memory issues"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @traceable(run_type="llm", name="Agent Routing Decision")
    def route_query(self, query: str) -> tuple[str, str]:
        """Route query to appropriate agent with strict LLM-based routing"""
        if self.model is None or self.tokenizer is None:
            return "doping", "[MOCK] Routing to doping agent"

        try:
            instruction = """You are an expert semiconductor processing orchestrator. You must analyze the query and respond in EXACTLY this format:

AGENT: [a single agent_name]
REASON: [brief_reason]

Where [agent_name] must be EXACTLY one of: doping, etching, wafer

Agent Selection Guide:
- doping: Ion implantation, dopant activation, junction formation, annealing, boron/phosphorus/arsenic processes, co-implantation, multiple energy implants, channeling effects, dopant segregation, concentration profiles, diffusion modeling
- etching: Plasma etching, chemical etching, pattern transfer, material removal, selectivity, RIE, ICP, gas chemistry, wet etching, etch rates, sidewall profiles, endpoint detection, plasma damage, aspect ratio dependent etching
- wafer: Wafer cleaning, oxidation, CVD, packaging, surface preparation, RCA cleaning, particle removal, thermal oxidation, LPCVD, PECVD, ALD, die attachment, wire bonding, CMP processes, defect inspection, metrology

Examples:

Query: I need boron implantation for p-type source/drain regions
AGENT: doping
REASON: Boron implantation is a doping process

Query: I need to etch silicon dioxide selectively over silicon nitride
AGENT: etching
REASON: Selective etching requires plasma etching expertise

Query: I need RCA cleaning before thermal oxidation
AGENT: wafer
REASON: RCA cleaning is wafer surface preparation

You MUST follow the exact format above with only a single agent and reason field. Do not add any other text."""

            alpaca_prompt = self.create_alpaca_prompt(
                instruction, f"Query: {query}")

            inputs = self.tokenizer(
                alpaca_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1536,
                padding=False,
            )

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if hasattr(self.model, 'past_key_values'):
                self.model.past_key_values = None

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]                           :], skip_special_tokens=True
            ).strip()

            self._clear_gpu_cache()

            print(f"[DEBUG] Orchestrator raw response: '{response}'")

            # Parse structured response
            agent_name = "doping"  # default
            reasoning = "Failed to parse response"

            # Look for AGENT: and REASON: patterns
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("AGENT:"):
                    agent_part = line[6:].strip().lower()
                    if agent_part in self.agents:
                        agent_name = agent_part
                elif line.startswith("REASON:"):
                    reasoning = line[7:].strip()

            # Fallback parsing if structured format not found
            if agent_name == "doping" and reasoning == "Failed to parse response":
                response_lower = response.lower()
                if "etching" in response_lower or "etch" in response_lower:
                    agent_name = "etching"
                    reasoning = "Detected etching-related content"
                elif "wafer" in response_lower or "clean" in response_lower or "oxidation" in response_lower:
                    agent_name = "wafer"
                    reasoning = "Detected wafer processing content"
                else:
                    reasoning = "Defaulted to doping agent"

            print(f"[DEBUG] Parsed - Agent: {agent_name}, Reason: {reasoning}")
            return agent_name, reasoning

        except Exception as e:
            print(f"[ERROR] Orchestrator: Error in routing: {e}")
            self._clear_gpu_cache()
            return "doping", f"[FALLBACK] Error in routing: {e}"

    @traceable(run_type="llm", name="Dependency Analysis")
    def analyze_dependencies(self, solution: str, current_agent: str) -> tuple[bool, str, Optional[str]]:
        """Analyze if the solution requires input from other agents using LLM with examples"""
        if self.model is None or self.tokenizer is None:
            return False, "[MOCK] No dependencies found", None

        try:
            other_agents = [
                agent for agent in self.agents.keys() if agent != current_agent]

            # UPDATED PROMPT - This is where Option 2 goes
            instruction = f"""You are analyzing a semiconductor processing solution to determine if it depends on parameters from OTHER processing steps.

    Current solution is from {current_agent} agent. DO NOT suggest the same agent as next agent.

    IMPORTANT: If the solution needs parameter adjustments within the SAME processing domain ({current_agent}), 
    respond with DEPENDENCIES: NO since no external agents are needed. Internal parameter optimization 
    will be handled through human feedback.

    Only respond with DEPENDENCIES: YES if you need parameters from a DIFFERENT processing domain.

    Other available agents (excluding current):
    {chr(10).join([f"- {agent}: {desc}" for agent, desc in [
        ("doping", "Ion implantation, dopant activation, junction formation, annealing processes"),
        ("etching", "Plasma etching, chemical etching, pattern transfer, material removal, selectivity"),
        ("wafer", "Wafer cleaning, oxidation, CVD, packaging, surface preparation, metrology")
    ] if agent != current_agent])}

    Respond in EXACTLY this format:

    If dependencies are found from OTHER agents:
    DEPENDENCIES: YES
    EXPLANATION: [brief explanation of what parameters are needed from other agents]
    NEXT_AGENT: [agent_name from OTHER agents, NOT {current_agent}]

    If no dependencies from other agents (including internal parameter optimization):
    DEPENDENCIES: NO
    EXPLANATION: [brief explanation why no external dependencies needed]

    Examples:

    DEPENDENCIES: YES
    EXPLANATION: Requires substrate preparation parameters from wafer processing
    NEXT_AGENT: wafer

    DEPENDENCIES: NO
    EXPLANATION: Solution uses only internal etching parameters, any optimization can be done through feedback

    DEPENDENCIES: NO
    EXPLANATION: All required doping parameters are specified, no external process dependencies

    You MUST follow the exact format above."""

            alpaca_prompt = self.create_alpaca_prompt(
                instruction, f"Solution to analyze:\n{solution}")

            # ... rest of the method remains the same
            inputs = self.tokenizer(
                alpaca_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1536,
                padding=False,
            )

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if hasattr(self.model, 'past_key_values'):
                self.model.past_key_values = None

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]                           :], skip_special_tokens=True
            ).strip()

            self._clear_gpu_cache()

            print(f"[DEBUG] Dependency analysis raw response: '{response}'")

            # Parse structured response with fixed indexing
            has_deps = False
            explanation = "No explanation provided"
            next_agent = None

            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("DEPENDENCIES:"):
                    deps_part = line[13:].strip().upper()
                    has_deps = deps_part == "YES"
                elif line.startswith("EXPLANATION:"):
                    explanation = line[12:].strip()
                elif line.startswith("NEXT_AGENT:"):
                    # Fixed: Use more robust parsing to handle the colon properly
                    agent_part = line.split(":", 1)[1].strip().lower()
                    if agent_part in self.agents and agent_part != current_agent:
                        next_agent = agent_part

            print(
                f"[DEBUG] Parsed - Dependencies: {has_deps}, Next: {next_agent}")
            return has_deps, explanation, next_agent

        except Exception as e:
            print(f"[ERROR] Orchestrator: Error in dependency analysis: {e}")
            self._clear_gpu_cache()
            return False, f"[FALLBACK] Error in analysis: {e}", None


# Global orchestrator instance
orchestrator = Orchestrator()


@traceable(run_type="chain", name="Orchestrator Node")
def orchestrator_node(state: MultiAgentState) -> MultiAgentState:
    """Orchestrator node that routes queries to appropriate agents"""
    print(f"[DEBUG] Orchestrator node started")
    if not state.get('selected_agent'):
        print(f"[DEBUG] Routing query: {state['user_query']}")
        # Initial routing
        agent_name, reasoning = orchestrator.route_query(state['user_query'])
        print(f"[DEBUG] Selected agent: {agent_name}, reasoning: {reasoning}")
        state['selected_agent'] = agent_name
        state['agent_reasoning'] = reasoning
        state['completed_agents'] = []
        state['agent_chain'] = [agent_name]

    print(f"[DEBUG] Orchestrator node completed")
    return state


@traceable(run_type="chain", name="Agent Processing Node")
def agent_processing_node(state: MultiAgentState) -> MultiAgentState:
    """Process query with the selected agent"""
    agent_name = state['selected_agent']
    agent = orchestrator.agents[agent_name]

    # Find relevant program if not already selected
    if not state.get('selected_program'):
        selected_program = agent.find_relevant_program(state['user_query'])
        state['selected_program'] = selected_program

    # Get previous solution and feedback
    feedback = state.get('feedback', '')
    previous_solution = state.get('llm_solution', '')

    # Generate solution
    solution = agent.generate_solution(
        state['user_query'],
        state['selected_program'],
        feedback,
        previous_solution
    )

    # Update state
    state['llm_solution'] = solution
    state['iteration_count'] = state.get('iteration_count', 0) + 1

    # Add current agent to completed list if not already there
    if agent_name not in state.get('completed_agents', []):
        completed = state.get('completed_agents', [])
        completed.append(agent_name)
        state['completed_agents'] = completed

    # Parse parameters from solution
    parameters = {
        'iteration': state['iteration_count'],
        'agent_used': agent_name,
        'program_used': state['selected_program'].get('task', 'Knowledge-based') if state['selected_program'] else 'Knowledge-based',
        'solution': solution
    }
    state['final_parameters'] = parameters

    return state


@traceable(run_type="chain", name="Dependency Analysis Node")
def dependency_analysis_node(state: MultiAgentState) -> Command[Literal["agent_processing", "human_feedback"]]:
    """Analyze dependencies and route to next agent if needed"""
    current_agent = state['selected_agent']
    solution = state['llm_solution']
    completed_agents = state.get('completed_agents', [])

    # Analyze dependencies
    has_deps, explanation, next_agent = orchestrator.analyze_dependencies(
        solution, current_agent)

    state['dependency_analysis'] = explanation
    state['requires_other_agents'] = has_deps

    if has_deps and next_agent and next_agent not in completed_agents:
        # Route to the next agent
        print(f"\n[ORCHESTRATOR] Dependency found: {explanation}")
        print(f"[ORCHESTRATOR] Routing to {next_agent} agent...")

        # Update state for next agent
        agent_chain = state.get('agent_chain', [])
        agent_chain.append(next_agent)

        return Command(
            goto="agent_processing",
            update={
                "selected_agent": next_agent,
                "selected_program": {},  # Reset program selection for new agent
                "agent_chain": agent_chain
            }
        )
    else:
        # No more dependencies, proceed to human feedback
        return Command(goto="human_feedback")


@traceable(run_type="tool", name="Human Feedback Collection")
def human_feedback_node(state: MultiAgentState) -> Command[Literal["continue_feedback", "finalize"]]:
    """Human feedback node for collecting user input"""

    agent_info = f"Agent Chain: {' → '.join(state.get('agent_chain', []))}"

    feedback_data = interrupt({
        "question": "Review the multi-agent solution. Do you want to make changes?",
        "solution": state['llm_solution'],
        "iteration": state['iteration_count'],
        "agent_info": agent_info,
        "dependency_analysis": state.get('dependency_analysis', 'No analysis available'),
        "options": [
            "approve - Accept current solution",
            "modify:<feedback> - Request changes (e.g. 'modify:reduce thermal budget by 10%')",
            "exit - Finalize current solution"
        ]
    })

    if feedback_data.startswith("modify:"):
        feedback = feedback_data[7:].strip()
        changes = state.get('change_summary', [])
        changes.append(f"Iteration {state['iteration_count']}: {feedback}")

        return Command(
            goto="continue_feedback",
            update={
                "feedback": feedback,
                "change_summary": changes
            }
        )
    else:
        return Command(goto="finalize")


@traceable(run_type="tool", name="Continue Feedback Processing")
def continue_feedback_node(state: MultiAgentState) -> Command[Literal["human_feedback"]]:
    """Process feedback and continue with human feedback loop"""
    # Reprocess with current agent based on feedback
    agent_name = state['selected_agent']
    agent = orchestrator.agents[agent_name]

    feedback = state.get('feedback', '')
    previous_solution = state.get('llm_solution', '')

    # ADD DEBUGGING
    print(f"[DEBUG] continue_feedback_node:")
    print(f"[DEBUG] Agent: {agent_name}")
    print(f"[DEBUG] Feedback: '{feedback}'")
    print(f"[DEBUG] Previous solution length: {len(previous_solution)} chars")

    # Generate updated solution
    solution = agent.generate_solution(
        state['user_query'],
        state['selected_program'],
        feedback,
        previous_solution
    )

    # MORE DEBUGGING
    print(f"[DEBUG] New solution length: {len(solution)} chars")
    print(f"[DEBUG] New solution preview: {solution[:200]}...")
    print(f"[DEBUG] Solutions are identical: {solution == previous_solution}")

    # Update iteration count
    new_iteration_count = state.get('iteration_count', 0) + 1

    # Create parameters dict
    parameters = {
        'iteration': new_iteration_count,
        'agent_used': agent_name,
        'program_used': state['selected_program'].get('task', 'Knowledge-based') if state['selected_program'] else 'Knowledge-based',
        'solution': solution
    }

    # FIXED: Use the update parameter in Command to ensure state is properly updated
    return Command(
        goto="human_feedback",
        update={
            'llm_solution': solution,
            'iteration_count': new_iteration_count,
            'final_parameters': parameters
        }
    )


@traceable(run_type="tool", name="Finalize Multi-Agent Solution")
def finalize_node(state: MultiAgentState) -> MultiAgentState:
    """Final node that presents the optimized multi-agent solution"""
    print("\n" + "="*60)
    print("FINAL OPTIMIZED MULTI-AGENT SOLUTION")
    print("="*60)
    print(f"Query: {state['user_query']}")
    print(f"Agent Chain: {' → '.join(state.get('agent_chain', []))}")
    print(f"Final Agent: {state['selected_agent']}")
    print(
        f"Program Used: {state['selected_program'].get('task', 'Knowledge-based') if state['selected_program'] else 'Knowledge-based'}")
    print(f"Total Iterations: {state['iteration_count']}")

    if state.get('dependency_analysis'):
        print(f"\nDependency Analysis: {state['dependency_analysis']}")

    if state.get('change_summary'):
        print("\nChanges Made:")
        for change in state['change_summary']:
            print(f"- {change}")

    print(f"\nFinal Solution:\n{state['llm_solution']}")
    print("="*60)

    return state


# Build the Multi-Agent LangGraph workflow
@traceable(run_type="chain", name="Create Multi-Agent Workflow")
def create_multi_agent_workflow():
    """Create and return the multi-agent workflow graph"""

    builder = StateGraph(MultiAgentState)

    # Add nodes
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("agent_processing", agent_processing_node)
    builder.add_node("dependency_analysis", dependency_analysis_node)
    builder.add_node("human_feedback", human_feedback_node)
    builder.add_node("continue_feedback", continue_feedback_node)
    builder.add_node("finalize", finalize_node)

    # Set entry point
    builder.set_entry_point("orchestrator")

    # Add edges
    builder.add_edge("orchestrator", "agent_processing")
    builder.add_edge("agent_processing", "dependency_analysis")
    # dependency_analysis uses Command to route to either "agent_processing" or "human_feedback"
    # human_feedback uses Command to route to either "continue_feedback" or "finalize"
    # CHANGED: was "dependency_analysis"
    builder.add_edge("continue_feedback", "human_feedback")
    builder.add_edge("finalize", END)

    # Compile with checkpointer for human-in-the-loop
    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


@traceable(run_type="chain", name="Run Multi-Agent Workflow")
def run_multi_agent_workflow(user_query: str):
    """Run the multi-agent workflow with a user query"""

    graph = create_multi_agent_workflow()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print(f"Starting multi-agent workflow for query: {user_query}")
    print("-" * 60)

    # Initial state
    initial_state = {
        "user_query": user_query,
        "iteration_count": 0,
        "change_summary": [],
        "completed_agents": [],
        "agent_chain": []
    }

    # Run until first interrupt (human feedback)
    result = graph.invoke(initial_state, config=config)

    # Handle human feedback loop
    while "__interrupt__" in result:
        interrupt_data = result["__interrupt__"]

        # Handle both list and dict formats for interrupt data
        if isinstance(interrupt_data, list) and len(interrupt_data) > 0:
            interrupt_value = interrupt_data[0]
            if hasattr(interrupt_value, 'value'):
                interrupt_dict = interrupt_value.value
            else:
                interrupt_dict = interrupt_value
        elif hasattr(interrupt_data, 'value'):
            interrupt_dict = interrupt_data.value
        else:
            interrupt_dict = interrupt_data

        print(f"\n{interrupt_dict['question']}")
        print(f"Agent Info: {interrupt_dict.get('agent_info', 'N/A')}")
        print(
            f"Dependency Analysis: {interrupt_dict.get('dependency_analysis', 'N/A')}")
        print(f"Current solution:\n{result['llm_solution']}")
        print("\nOptions:")
        for option in interrupt_dict["options"]:
            print(f"  - {option}")

        # Get user input
        user_input = input("\nYour choice: ").strip()

        # Resume workflow with user input
        result = graph.invoke(Command(resume=user_input), config=config)

    return result


if __name__ == "__main__":
    # Example usage
    query = "How to etch silicon dioxide using CHF3 plasma chemistry with anisotropic profile control?"
    result = run_multi_agent_workflow(query)
