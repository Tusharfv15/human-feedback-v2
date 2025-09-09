"""
Streamlit Interface for Multi-Agent Semiconductor Processing Workflow
"""

import streamlit as st
import uuid
import json
from typing import Dict, Any, List
import sys
import os

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multi_agent_workflow import (
    create_multi_agent_workflow,
    MultiAgentState,
    orchestrator
)
from langgraph.types import Command

# Page configuration
st.set_page_config(
    page_title="Semiconductor Processing Multi-Agent Workflow",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .agent-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
    }
    .solution-box {
        border: 1px solid #28a745;
        border-radius: 8px;
        padding: 1rem;
        background-color: #f8fff8;
        margin: 1rem 0;
    }
    .feedback-box {
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        background-color: #fffef0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'workflow_state' not in st.session_state:
    st.session_state.workflow_state = None
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'config' not in st.session_state:
    st.session_state.config = None
if 'workflow_history' not in st.session_state:
    st.session_state.workflow_history = []
if 'awaiting_feedback' not in st.session_state:
    st.session_state.awaiting_feedback = False
if 'interrupt_data' not in st.session_state:
    st.session_state.interrupt_data = None

def initialize_workflow():
    """Initialize the workflow graph and configuration"""
    if st.session_state.graph is None:
        st.session_state.graph = create_multi_agent_workflow()
        st.session_state.config = {"configurable": {"thread_id": str(uuid.uuid4())}}

def display_agent_info():
    """Display information about available agents"""
    st.sidebar.header("🤖 Available Agents")
    
    agents_info = {
        "Doping Agent": {
            "description": "Ion implantation, dopant activation, junction formation, annealing",
            "specialties": ["Boron/Phosphorus/Arsenic processes", "Co-implantation", "Multiple energy implants", "Channeling effects"]
        },
        "Etching Agent": {
            "description": "Plasma etching, chemical etching, pattern transfer, material removal",
            "specialties": ["RIE/ICP processing", "Gas chemistry", "Wet etching", "Selectivity optimization"]
        },
        "Wafer Agent": {
            "description": "Wafer cleaning, oxidation, CVD, packaging, surface preparation",
            "specialties": ["RCA cleaning", "Thermal oxidation", "LPCVD/PECVD/ALD", "CMP processes"]
        }
    }
    
    for agent_name, info in agents_info.items():
        with st.sidebar.expander(f"{agent_name}"):
            st.write(f"**Main Focus:** {info['description']}")
            st.write("**Specialties:**")
            for specialty in info['specialties']:
                st.write(f"• {specialty}")

def display_workflow_status():
    """Display current workflow status"""
    if st.session_state.workflow_state:
        state = st.session_state.workflow_state
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Agent", state.get('selected_agent', 'None'))
        
        with col2:
            st.metric("Iteration Count", state.get('iteration_count', 0))
        
        with col3:
            completed_count = len(state.get('completed_agents', []))
            st.metric("Completed Agents", completed_count)
        
        # Display agent chain
        if state.get('agent_chain'):
            st.info(f"**Agent Chain:** {' → '.join(state['agent_chain'])}")

def display_solution(state: Dict[str, Any]):
    """Display the current solution"""
    if state.get('llm_solution'):
        st.markdown('<div class="solution-box">', unsafe_allow_html=True)
        st.subheader("💡 Current Solution")
        st.write(state['llm_solution'])
        
        # Display additional information
        if state.get('selected_program') and state['selected_program']:
            st.write(f"**Program Used:** {state['selected_program'].get('task', 'Knowledge-based')}")
        else:
            st.write("**Program Used:** Knowledge-based")
            
        if state.get('dependency_analysis'):
            st.write(f"**Dependency Analysis:** {state['dependency_analysis']}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def handle_feedback_interface():
    """Handle the feedback interface when workflow is interrupted"""
    if st.session_state.interrupt_data:
        interrupt_dict = st.session_state.interrupt_data
        
        st.markdown('<div class="feedback-box">', unsafe_allow_html=True)
        st.subheader("🔄 Human Feedback Required")
        st.write(interrupt_dict.get('question', 'Please review the solution'))
        
        if interrupt_dict.get('agent_info'):
            st.info(f"**Agent Info:** {interrupt_dict['agent_info']}")
        
        if interrupt_dict.get('dependency_analysis'):
            st.info(f"**Dependency Analysis:** {interrupt_dict['dependency_analysis']}")
        
        # Display current solution
        if st.session_state.workflow_state and st.session_state.workflow_state.get('llm_solution'):
            with st.expander("View Current Solution", expanded=True):
                st.write(st.session_state.workflow_state['llm_solution'])
        
        st.subheader("Choose Your Action:")
        
        # Action selection
        action = st.radio(
            "Select an action:",
            ["Approve Solution", "Request Modifications", "Exit and Finalize"],
            key="feedback_action"
        )
        
        feedback_input = ""
        if action == "Request Modifications":
            feedback_input = st.text_area(
                "Enter your feedback/modifications:",
                placeholder="e.g., 'reduce thermal budget by 10%', 'increase selectivity to 50:1', 'adjust implant energy to 15 keV'",
                key="modification_text"
            )
        
        if st.button("Submit Response", type="primary"):
            if action == "Approve Solution":
                user_response = "approve"
            elif action == "Request Modifications":
                if feedback_input.strip():
                    user_response = f"modify:{feedback_input.strip()}"
                else:
                    st.error("Please enter your modification request.")
                    return
            else:  # Exit and Finalize
                user_response = "exit"
            
            # Continue workflow with user input
            try:
                result = st.session_state.graph.invoke(
                    Command(resume=user_response), 
                    config=st.session_state.config
                )
                
                # Update session state
                st.session_state.workflow_state = result
                
                # Check if still awaiting feedback
                if "__interrupt__" in result:
                    st.session_state.awaiting_feedback = True
                    st.session_state.interrupt_data = parse_interrupt_data(result["__interrupt__"])
                else:
                    st.session_state.awaiting_feedback = False
                    st.session_state.interrupt_data = None
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing feedback: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def parse_interrupt_data(interrupt_data):
    """Parse interrupt data from different formats"""
    if isinstance(interrupt_data, list) and len(interrupt_data) > 0:
        interrupt_value = interrupt_data[0]
        if hasattr(interrupt_value, 'value'):
            return interrupt_value.value
        else:
            return interrupt_value
    elif hasattr(interrupt_data, 'value'):
        return interrupt_data.value
    else:
        return interrupt_data

def display_workflow_history():
    """Display workflow execution history"""
    if st.session_state.workflow_history:
        with st.expander("📜 Workflow History"):
            for i, entry in enumerate(st.session_state.workflow_history):
                st.write(f"**Step {i+1}:** {entry}")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔬 Semiconductor Processing Multi-Agent Workflow</h1>', unsafe_allow_html=True)
    
    # Initialize workflow
    initialize_workflow()
    
    # Sidebar with agent information
    display_agent_info()
    
    # Main interface
    if not st.session_state.awaiting_feedback:
        # Query input section
        st.header("🎯 Enter Your Semiconductor Processing Query")
        
        # Example queries
        st.subheader("💡 Example Queries")
        examples = [
            "How to etch silicon dioxide using CHF3 plasma chemistry with anisotropic profile control?",
            "I need boron implantation for p-type source/drain regions with 50nm junction depth",
            "RCA cleaning procedure before thermal oxidation for 10nm gate oxide",
            "Phosphorus implantation followed by rapid thermal annealing for n+ contact formation",
            "Selective etching of silicon nitride over silicon dioxide with 20:1 selectivity"
        ]
        
        selected_example = st.selectbox("Choose an example or enter your own:", [""] + examples)
        
        # Query input
        user_query = st.text_area(
            "Enter your query:",
            value=selected_example if selected_example else "",
            height=100,
            placeholder="Describe your semiconductor processing requirement..."
        )
        
        # Start workflow button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🚀 Start Workflow", type="primary", disabled=not user_query.strip()):
                if user_query.strip():
                    try:
                        # Reset session state for new workflow
                        st.session_state.workflow_history = []
                        st.session_state.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
                        
                        # Initial state
                        initial_state = {
                            "user_query": user_query.strip(),
                            "iteration_count": 0,
                            "change_summary": [],
                            "completed_agents": [],
                            "agent_chain": []
                        }
                        
                        # Start workflow
                        with st.spinner("Processing your query..."):
                            result = st.session_state.graph.invoke(initial_state, config=st.session_state.config)
                        
                        # Update session state
                        st.session_state.workflow_state = result
                        
                        # Check if awaiting feedback
                        if "__interrupt__" in result:
                            st.session_state.awaiting_feedback = True
                            st.session_state.interrupt_data = parse_interrupt_data(result["__interrupt__"])
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error starting workflow: {str(e)}")
                        st.exception(e)
        
        with col2:
            if st.button("🔄 Reset Workflow"):
                # Clear session state
                for key in ['workflow_state', 'workflow_history', 'awaiting_feedback', 'interrupt_data']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
                st.rerun()
    
    else:
        # Handle feedback interface
        handle_feedback_interface()
    
    # Display current workflow status
    if st.session_state.workflow_state:
        st.header("📊 Workflow Status")
        display_workflow_status()
        
        # Display current solution
        display_solution(st.session_state.workflow_state)
        
        # Display change summary if available
        if st.session_state.workflow_state.get('change_summary'):
            with st.expander("📝 Change History"):
                for change in st.session_state.workflow_state['change_summary']:
                    st.write(f"• {change}")
    
    # Display workflow history
    display_workflow_history()
    
    # Footer
    st.markdown("---")
    st.markdown("*Multi-Agent Semiconductor Processing Workflow - Powered by LangGraph*")

if __name__ == "__main__":
    main()