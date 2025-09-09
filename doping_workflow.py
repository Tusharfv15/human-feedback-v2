"""
Simple LangGraph workflow for doping-related queries with human feedback loop.
"""

import json
import uuid
from typing import TypedDict, Literal, List, Dict, Any
import torch

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import traceable
from dotenv import load_dotenv
from model_manager import model_manager
load_dotenv()


class DopingState(TypedDict):
    """State for the doping workflow"""
    user_query: str
    selected_program: Dict[str, Any]
    llm_solution: str
    feedback: str
    iteration_count: int
    final_parameters: Dict[str, Any]
    change_summary: List[str]
    previous_solution: str  # New: stores the last generated solution


class DopingAgent:
    """Single agent that handles doping queries and feedback"""

    def __init__(self):
        self.model, self.tokenizer = model_manager.get_model()
        self.programs = self._load_programs()
        self.agent_name = "doping-agent"

    def _load_programs(self) -> Dict[str, Any]:
        """Load doping programs from JSON file"""
        with open('programs/doping-programs.json', 'r') as f:
            return json.load(f)

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
        return f"[FALLBACK] Unable to process: {prompt[:100]}..."

    @traceable(run_type="tool", name="LLM Program Search")
    def _llm_program_search(self, query: str) -> Dict[str, Any]:
        """Find the most relevant doping program using the finetuned LLM"""
        if self.model is None or self.tokenizer is None:
            return list(self.programs.values())[0]

        try:
            # Create instruction for program selection
            program_list = ""
            for program_id, program_data in self.programs.items():
                program_list += f"\n{program_id}: {program_data['task']}\n"
                for sub_htp in program_data.get('sub_htps', []):
                    program_list += f"  - {sub_htp['task']}\n"

            instruction = f"You are a specialized doping expert in semiconductor processing. Select the most relevant doping program for the given query. Respond with only the program_id (e.g., 'program_1', 'program_2', or 'program_3').\n\nAvailable Programs:{program_list}"

            # Create Alpaca format prompt
            alpaca_prompt = self.create_alpaca_prompt(instruction, query)

            # Tokenize input
            inputs = self.tokenizer(
                alpaca_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1536,
                padding=False,
            )

            # Move to correct device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if hasattr(self.model, 'past_key_values'):
                self.model.past_key_values = None

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    use_cache=False,
                )

            # Extract only the generated part
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]
                    :], skip_special_tokens=True
            ).strip()

            # Clear GPU cache
            self._clear_gpu_cache()

            # Extract program_id from response
            if response in self.programs:
                return self.programs[response]
            else:
                # Fallback: find program_id in response text
                for program_id in self.programs.keys():
                    if program_id in response.lower():
                        return self.programs[program_id]

                # Final fallback to first program
                return list(self.programs.values())[0]

        except Exception as e:
            print(f"[ERROR] {self.agent_name}: Error in program search: {e}")
            self._clear_gpu_cache()
            return list(self.programs.values())[0]

    @traceable(run_type="llm", name="Generate Doping Solution")
    def _generate_solution(self, query: str, program: Dict[str, Any], feedback: str = None, previous_solution: str = None) -> str:
        """Generate solution using the finetuned LLM with proper Alpaca format"""
        if self.model is None or self.tokenizer is None:
            return f"[MOCK RESPONSE from {self.agent_name}] Processing: {query[:100]}..."

        try:
            # Create instruction for parameter generation
            program_details = f"Program: {program['task']}\n\nAvailable parameters and ranges:\n"
            for sub_htp in program.get('sub_htps', []):
                program_details += f"\nTask: {sub_htp['task']}\n"
                for param in sub_htp.get('table', []):
                    program_details += f"- {param['parameter']}: {param['typical_range']} {param['units']}\n"
                if 'note' in sub_htp:
                    program_details += f"Note: {sub_htp['note']}\n"

            instruction = f"You are a specialized doping expert in semiconductor processing. Based on the program data provided, give specific parameter recommendations with explanations.\n\n{program_details}\n\nProvide specific parameter values with explanations:"

            # Combine query, previous solution, and feedback as input
            user_input = f"User Query: {query}"

            if previous_solution and feedback:
                user_input += f"\n\nPrevious Solution:\n{previous_solution}"
                user_input += f"\n\nUser Feedback: {feedback}\nPlease modify the previous solution based on the feedback."
            elif feedback:
                user_input += f"\n\nUser Feedback: {feedback}\nPlease adjust the recommendations based on the feedback."

            # Create Alpaca format prompt
            alpaca_prompt = self.create_alpaca_prompt(instruction, user_input)

            # Tokenize input
            inputs = self.tokenizer(
                alpaca_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1536,
                padding=False,
            )

            # Move to correct device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if hasattr(self.model, 'past_key_values'):
                self.model.past_key_values = None

            # Generate response with proper parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    top_p=0.9,
                    use_cache=False,
                )

            # Extract only the generated part
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]
                    :], skip_special_tokens=True
            ).strip()

            # Clear GPU cache
            self._clear_gpu_cache()

            # Debug logging
            print(
                f"[DEBUG] {self.agent_name}: Generated response length: {len(response)}")
            print(
                f"[DEBUG] {self.agent_name}: Response preview: {response[:100]}...")

            # Validation
            if not response or len(response.strip()) < 1:
                print(
                    f"[WARNING] {self.agent_name}: Generated response too short, using fallback")
                return self._generate_fallback_response(query)

            return response

        except Exception as e:
            print(f"[ERROR] {self.agent_name}: Error generating response: {e}")
            import traceback
            traceback.print_exc()
            self._clear_gpu_cache()
            return self._generate_fallback_response(query)


@traceable(run_type="chain", name="Doping Agent Node")
def doping_agent_node(state: DopingState) -> DopingState:
      """Main doping agent node that processes queries and generates solutions"""
    agent = DopingAgent()
    
    # Step 1: LLM-based program search for relevant program
    if not state.get('selected_program'):
        selected_program = agent._llm_program_search(state['user_query'])
        state['selected_program'] = selected_program
    
    # Step 2: Get previous solution BEFORE generating new one
    feedback = state.get('feedback', '')
    previous_solution = state.get('llm_solution', '')  # Get the last generated solution
    
    # Step 3: Generate new solution
    solution = agent._generate_solution(
        state['user_query'], 
        state['selected_program'], 
        feedback,
        previous_solution  # This will be empty on first run, populated on subsequent runs
    )
    
    # Step 4: Update state
    state['llm_solution'] = solution  # Store new solution
    state['iteration_count'] = state.get('iteration_count', 0) + 1
    
    # Parse parameters from solution (simplified)
    # In a real implementation, you'd have more sophisticated parameter extraction
    parameters = {
        'iteration': state['iteration_count'],
        'program_used': state['selected_program']['task'],
        'solution': solution
    }
    state['final_parameters'] = parameters
    
    return state


@traceable(run_type="tool", name="Human Feedback Collection")
def human_feedback_node(state: DopingState) -> Command[Literal["continue_feedback", "finalize"]]:
    """Human feedback node for collecting user input"""
    
    feedback_data = interrupt({
        "question": "Review the doping solution. Do you want to make changes?",
        "solution": state['llm_solution'],
        "iteration": state['iteration_count'],
        "options": [
            "approve - Accept current solution",
            "modify:<feedback> - Request changes (e.g. 'modify:reduce thermal budget by 10%')",
            "exit - Finalize current solution"
        ]
    })
    
    if feedback_data.startswith("modify:"):
        feedback = feedback_data[7:].strip()  # Remove "modify:" prefix
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
        # Either "approve" or "exit" - finalize the solution
        return Command(goto="finalize")


@traceable(run_type="tool", name="Finalize Solution")
def finalize_node(state: DopingState) -> DopingState:
    """Final node that presents the optimized solution"""
    print("\n" + "="*50)
    print("FINAL OPTIMIZED DOPING SOLUTION")
    print("="*50)
    print(f"Query: {state['user_query']}")
    print(f"Program Used: {state['selected_program']['task']}")
    print(f"Total Iterations: {state['iteration_count']}")
    
    if state.get('change_summary'):
        print("\nChanges Made:")
        for change in state['change_summary']:
            print(f"- {change}")
    
    print(f"\nFinal Solution:\n{state['llm_solution']}")
    print("="*50)
    
    return state


# Build the LangGraph workflow
@traceable(run_type="chain", name="Create Doping Workflow")
def create_doping_workflow():
    """Create and return the doping workflow graph"""
    
    builder = StateGraph(DopingState)
    
    # Add nodes
    builder.add_node("doping_agent", doping_agent_node)
    builder.add_node("human_feedback", human_feedback_node)
    builder.add_node("continue_feedback", doping_agent_node)  # Reuse agent for feedback iterations
    builder.add_node("finalize", finalize_node)
    
    # Set entry point
    builder.set_entry_point("doping_agent")
    
    # Add edges
    builder.add_edge("doping_agent", "human_feedback")
    builder.add_edge("continue_feedback", "human_feedback")
    builder.add_edge("finalize", END)
    
    # Compile with checkpointer for human-in-the-loop
    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


@traceable(run_type="chain", name="Run Doping Workflow")
def run_doping_workflow(user_query: str):
    """Run the doping workflow with a user query"""
    
    graph = create_doping_workflow()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    print(f"Starting doping workflow for query: {user_query}")
    print("-" * 50)
    
    # Initial state
    initial_state = {
        "user_query": user_query,
        "iteration_count": 0,
        "change_summary": []
    }
    
    # Run until first interrupt (human feedback)
    result = graph.invoke(initial_state, config=config)
    
    # Handle human feedback loop
    while "__interrupt__" in result:
        interrupt_data = result["__interrupt__"]
        
        # Handle both list and dict formats for interrupt data
        if isinstance(interrupt_data, list) and len(interrupt_data) > 0:
            # If it's a list, take the first element
            interrupt_value = interrupt_data[0]
            if hasattr(interrupt_value, 'value'):
                interrupt_dict = interrupt_value.value
            else:
                interrupt_dict = interrupt_value
        elif hasattr(interrupt_data, 'value'):
            # If it's an interrupt object with value attribute
            interrupt_dict = interrupt_data.value
        else:
            # If it's already a dictionary
            interrupt_dict = interrupt_data
        
        print("\n" + interrupt_dict["question"])
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
    query = "I need to create shallow p+ junctions for a CMOS process with minimal thermal budget"
    result = run_doping_workflow(query)