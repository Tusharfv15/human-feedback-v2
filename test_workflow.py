"""
Test the workflow step by step
"""

import torch
import traceback

def test_workflow_creation():
    """Test workflow graph creation"""
    print("Testing workflow creation...")
    try:
        from multi_agent_workflow import create_multi_agent_workflow
        graph = create_multi_agent_workflow()
        print("Workflow graph created successfully")
        return graph
    except Exception as e:
        print(f"Error creating workflow: {e}")
        traceback.print_exc()
        return None

def test_workflow_invoke():
    """Test workflow invocation"""
    print("Testing workflow invocation...")
    try:
        import uuid
        from multi_agent_workflow import create_multi_agent_workflow
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        graph = create_multi_agent_workflow()
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        
        initial_state = {
            "user_query": "I need boron implantation for shallow junctions",
            "iteration_count": 0,
            "change_summary": [],
            "completed_agents": [],
            "agent_chain": []
        }
        
        print("Starting workflow invocation...")
        result = graph.invoke(initial_state, config=config)
        print(f"Workflow result keys: {result.keys()}")
        print("Workflow invocation successful")
        return result
        
    except Exception as e:
        print(f"Error invoking workflow: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("="*50)
    print("TESTING WORKFLOW EXECUTION")
    print("="*50)
    
    # Test 1: Create workflow
    graph = test_workflow_creation()
    if not graph:
        print("Failed to create workflow - stopping tests")
        exit(1)
    
    print("-"*30)
    
    # Test 2: Invoke workflow
    result = test_workflow_invoke()
    if result:
        print("SUCCESS: Workflow executed without errors")
    else:
        print("FAILED: Workflow execution failed")