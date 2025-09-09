"""
Gradio Web Interface for Multi-Agent Semiconductor Processing Workflow
"""

import gradio as gr
import threading
import queue
import time
import json
from typing import Dict, Any, List, Tuple
from multi_agent_workflow import run_multi_agent_workflow, create_multi_agent_workflow
from langgraph.types import Command
import uuid


class WorkflowManager:
    """Manages the workflow execution and state for Gradio interface"""
    
    def __init__(self):
        self.current_workflow = None
        self.current_config = None
        self.workflow_state = None
        self.workflow_thread = None
        self.response_queue = queue.Queue()
        self.user_input_queue = queue.Queue()
        self.is_waiting_for_input = False
        self.workflow_complete = False
        
    def start_workflow(self, query: str) -> Tuple[str, str, bool]:
        """Start a new workflow with the given query"""
        try:
            # Reset state
            self.workflow_complete = False
            self.is_waiting_for_input = False
            
            # Clear queues
            while not self.response_queue.empty():
                try:
                    self.response_queue.get_nowait()
                except queue.Empty:
                    break
                    
            while not self.user_input_queue.empty():
                try:
                    self.user_input_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Create workflow
            self.current_workflow = create_multi_agent_workflow()
            self.current_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            
            # Initial state
            initial_state = {
                "user_query": query,
                "iteration_count": 0,
                "change_summary": [],
                "completed_agents": [],
                "agent_chain": []
            }
            
            # Start workflow in background thread
            self.workflow_thread = threading.Thread(
                target=self._run_workflow_background,
                args=(initial_state,),
                daemon=True
            )
            self.workflow_thread.start()
            
            return f"🚀 Started workflow for query: {query}", "", False
            
        except Exception as e:
            return f"❌ Error starting workflow: {e}", "", False
    
    def _run_workflow_background(self, initial_state: Dict[str, Any]):
        """Run workflow in background thread"""
        try:
            # Run until first interrupt
            result = self.current_workflow.invoke(initial_state, config=self.current_config)
            
            # Handle interrupts
            while "__interrupt__" in result and not self.workflow_complete:
                interrupt_data = result["__interrupt__"]
                
                # Parse interrupt data
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
                
                # Format solution display
                solution_display = self._format_solution_display(result, interrupt_dict)
                
                # Put response in queue and wait for user input
                self.response_queue.put({
                    'type': 'feedback_request',
                    'solution': solution_display,
                    'raw_result': result
                })
                
                self.is_waiting_for_input = True
                
                # Wait for user input
                user_input = self.user_input_queue.get()  # This blocks until input is provided
                
                if user_input == "STOP_WORKFLOW":
                    break
                    
                self.is_waiting_for_input = False
                
                # Continue workflow with user input
                result = self.current_workflow.invoke(Command(resume=user_input), config=self.current_config)
            
            # Workflow completed
            if not self.workflow_complete:
                final_display = self._format_final_display(result)
                self.response_queue.put({
                    'type': 'final_result',
                    'solution': final_display,
                    'raw_result': result
                })
                self.workflow_complete = True
                
        except Exception as e:
            self.response_queue.put({
                'type': 'error',
                'solution': f"❌ Workflow error: {e}",
                'raw_result': None
            })
            self.workflow_complete = True
    
    def _format_solution_display(self, result: Dict[str, Any], interrupt_dict: Dict[str, Any]) -> str:
        """Format the solution for display in Gradio"""
        agent_chain = " → ".join(result.get('agent_chain', []))
        dependency_analysis = result.get('dependency_analysis', 'No analysis available')
        
        display = f"""## 🔄 Workflow Status: Awaiting Feedback

**Agent Chain:** {agent_chain}  
**Current Agent:** {result.get('selected_agent', 'Unknown')}  
**Iteration:** {result.get('iteration_count', 0)}  
**Dependency Analysis:** {dependency_analysis}

### 💡 Current Solution:
{result.get('llm_solution', 'No solution generated')}

---
**Please provide feedback or approve the solution below.**
"""
        return display
    
    def _format_final_display(self, result: Dict[str, Any]) -> str:
        """Format the final result for display"""
        agent_chain = " → ".join(result.get('agent_chain', []))
        
        display = f"""## ✅ Workflow Complete!

**Query:** {result.get('user_query', 'Unknown')}  
**Agent Chain:** {agent_chain}  
**Final Agent:** {result.get('selected_agent', 'Unknown')}  
**Total Iterations:** {result.get('iteration_count', 0)}

"""
        
        if result.get('dependency_analysis'):
            display += f"**Dependency Analysis:** {result.get('dependency_analysis')}\n\n"
        
        if result.get('change_summary'):
            display += "### 📝 Changes Made:\n"
            for change in result['change_summary']:
                display += f"- {change}\n"
            display += "\n"
        
        display += f"""### 🎯 Final Solution:
{result.get('llm_solution', 'No solution available')}
"""
        return display
    
    def provide_feedback(self, feedback_text: str) -> str:
        """Provide feedback to the workflow"""
        if not self.is_waiting_for_input:
            return "⚠️ No workflow is currently waiting for input."
        
        if not feedback_text.strip():
            return "⚠️ Please provide feedback text."
        
        # Format feedback
        if feedback_text.lower() in ['approve', 'exit']:
            user_input = feedback_text.lower()
        else:
            user_input = f"modify:{feedback_text}"
        
        # Send to workflow
        self.user_input_queue.put(user_input)
        
        return f"✅ Feedback submitted: {feedback_text}"
    
    def approve_solution(self) -> str:
        """Approve the current solution"""
        if not self.is_waiting_for_input:
            return "⚠️ No workflow is currently waiting for input."
        
        self.user_input_queue.put("approve")
        return "✅ Solution approved!"
    
    def get_workflow_update(self) -> Tuple[str, bool, bool]:
        """Get the latest workflow update"""
        try:
            response = self.response_queue.get_nowait()
            
            if response['type'] == 'feedback_request':
                return response['solution'], True, False  # solution, waiting_for_input, complete
            elif response['type'] == 'final_result':
                return response['solution'], False, True  # solution, waiting_for_input, complete
            elif response['type'] == 'error':
                return response['solution'], False, True  # solution, waiting_for_input, complete
                
        except queue.Empty:
            pass
        
        return "", self.is_waiting_for_input, self.workflow_complete


# Global workflow manager
workflow_manager = WorkflowManager()


def start_new_workflow(query: str) -> Tuple[str, str, str]:
    """Start a new workflow"""
    if not query.strip():
        return "⚠️ Please enter a query.", "", "disabled"
    
    result, _, _ = workflow_manager.start_workflow(query)
    return result, "", "disabled"


def submit_feedback(feedback: str) -> str:
    """Submit feedback to the workflow"""
    return workflow_manager.provide_feedback(feedback)


def approve_current_solution() -> str:
    """Approve the current solution"""
    return workflow_manager.approve_solution()


def update_workflow_display() -> Tuple[str, str, str, str]:
    """Update the workflow display"""
    solution, waiting_for_input, complete = workflow_manager.get_workflow_update()
    
    if solution:
        # Update feedback section visibility
        feedback_visibility = "visible" if waiting_for_input else "hidden"
        start_button_state = "enabled" if complete else "disabled"
        
        return solution, "", feedback_visibility, start_button_state
    
    # No update available
    current_visibility = "visible" if workflow_manager.is_waiting_for_input else "hidden"
    current_button_state = "enabled" if workflow_manager.workflow_complete else "disabled"
    
    return gr.update(), "", current_visibility, current_button_state


# Create Gradio interface
def create_gradio_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Multi-Agent Semiconductor Processing Workflow", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("""
        # 🔬 Multi-Agent Semiconductor Processing Workflow
        
        This interface allows you to interact with a multi-agent system for semiconductor processing queries. 
        The system will route your query to the appropriate specialized agent (Doping, Etching, or Wafer) 
        and provide expert recommendations.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Query Input Section
                gr.Markdown("## 📝 Enter Your Query")
                query_input = gr.Textbox(
                    label="Semiconductor Processing Query",
                    placeholder="e.g., I need boron implantation for shallow p+ junctions with minimal thermal budget",
                    lines=3,
                    max_lines=5
                )
                
                start_button = gr.Button("🚀 Start Workflow", variant="primary")
                start_status = gr.Textbox(label="Status", interactive=False)
                
            with gr.Column(scale=1):
                # Example Queries
                gr.Markdown("## 💡 Example Queries")
                example_queries = [
                    "I need boron implantation for shallow p+ junctions",
                    "How to etch silicon dioxide selectively over silicon nitride?", 
                    "I need RCA cleaning before thermal oxidation",
                    "Complete CMOS fabrication with shallow junctions and selective etching"
                ]
                
                for i, example in enumerate(example_queries):
                    gr.Button(f"Example {i+1}: {example[:40]}...", size="sm").click(
                        lambda ex=example: ex,
                        outputs=query_input
                    )
        
        # Workflow Display Section
        gr.Markdown("## 🔄 Workflow Progress")
        workflow_display = gr.Markdown("No workflow started yet.")
        
        # Feedback Section (initially hidden)
        with gr.Group(visible=False) as feedback_section:
            gr.Markdown("### 💭 Provide Feedback")
            
            with gr.Row():
                feedback_input = gr.Textbox(
                    label="Feedback/Modifications",
                    placeholder="e.g., reduce thermal budget by 20%, increase selectivity, use different chemistry",
                    lines=2,
                    scale=3
                )
                with gr.Column(scale=1):
                    submit_feedback_btn = gr.Button("📤 Submit Feedback", variant="secondary")
                    approve_btn = gr.Button("✅ Approve Solution", variant="primary")
        
        feedback_status = gr.Textbox(label="Feedback Status", interactive=False, visible=False)
        
        # Event handlers
        start_button.click(
            start_new_workflow,
            inputs=[query_input],
            outputs=[start_status, feedback_input, start_button]
        )
        
        submit_feedback_btn.click(
            submit_feedback,
            inputs=[feedback_input],
            outputs=[feedback_status]
        ).then(
            lambda: "",
            outputs=[feedback_input]
        )
        
        approve_btn.click(
            approve_current_solution,
            outputs=[feedback_status]
        )
        
        # Manual refresh button (simpler and more reliable)
        refresh_button = gr.Button("🔄 Refresh Status", size="sm")
        refresh_button.click(
            update_workflow_display,
            outputs=[workflow_display, feedback_status, feedback_section, start_button]
        )
        
        # Additional info
        with gr.Accordion("ℹ️ How to Use", open=False):
            gr.Markdown("""
            1. **Enter your query** about semiconductor processing (doping, etching, or wafer processing)
            2. **Click "Start Workflow"** to begin the multi-agent analysis
            3. **Review the solution** generated by the appropriate agent(s)
            4. **Provide feedback** if you want modifications, or **approve** if satisfied
            5. **Iterate** until you get the optimal solution
            
            **Agent Capabilities:**
            - **Doping Agent**: Ion implantation, dopant activation, junction formation
            - **Etching Agent**: Plasma etching, pattern transfer, selectivity control  
            - **Wafer Agent**: Surface preparation, cleaning, oxidation, packaging
            """)
        
        with gr.Accordion("🔧 Technical Details", open=False):
            gr.Markdown("""
            **Multi-Agent Architecture:**
            - LLM-based orchestrator routes queries to specialized agents
            - Dependency analysis determines if multiple agents are needed
            - Human-in-the-loop feedback for iterative refinement
            - Structured program selection with knowledge base fallback
            
            **Workflow Steps:**
            1. Query routing → Agent selection
            2. Program search → Solution generation  
            3. Dependency analysis → Multi-agent chaining (if needed)
            4. Human feedback → Solution refinement
            5. Final optimization → Complete solution
            """)
    
    return interface


if __name__ == "__main__":
    # Create and launch the interface
    interface = create_gradio_interface()
    interface.launch(
        server_name="127.0.0.1",  # Local access only
        server_port=7860,         # Default Gradio port
        share=False,              # Set to True to create public link
        debug=True,               # Enable debug mode
        show_error=True          # Show detailed errors
    )