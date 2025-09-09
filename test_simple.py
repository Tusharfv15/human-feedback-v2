"""
Simple test to debug the multi-agent workflow
"""

import torch
from model_manager import model_manager

def test_model_loading():
    """Test if model loads correctly"""
    print("Testing model loading...")
    try:
        model, tokenizer = model_manager.get_model()
        print(f"Model loaded: {model is not None}")
        print(f"Tokenizer loaded: {tokenizer is not None}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def test_orchestrator():
    """Test orchestrator creation"""
    print("Testing orchestrator...")
    try:
        from multi_agent_workflow import Orchestrator
        orchestrator = Orchestrator()
        print("Orchestrator created successfully")
        return True
    except Exception as e:
        print(f"Error creating orchestrator: {e}")
        return False

def test_agents():
    """Test agent creation"""
    print("Testing agents...")
    try:
        from multi_agent_workflow import DopingAgent, EtchingAgent, WaferAgent
        
        doping_agent = DopingAgent()
        print("Doping agent created")
        
        etching_agent = EtchingAgent()
        print("Etching agent created")
        
        wafer_agent = WaferAgent()
        print("Wafer agent created")
        
        return True
    except Exception as e:
        print(f"Error creating agents: {e}")
        return False

def test_simple_routing():
    """Test simple query routing"""
    print("Testing simple routing...")
    try:
        from multi_agent_workflow import orchestrator
        query = "I need boron implantation"
        agent_name, reasoning = orchestrator.route_query(query)
        print(f"Routed to: {agent_name}")
        print(f"Reasoning: {reasoning}")
        return True
    except Exception as e:
        print(f"Error in routing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*50)
    print("DEBUGGING MULTI-AGENT WORKFLOW")
    print("="*50)
    
    # Clear GPU cache first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    success = True
    success &= test_model_loading()
    print("-"*30)
    success &= test_orchestrator()
    print("-"*30)
    success &= test_agents()
    print("-"*30)
    success &= test_simple_routing()
    
    print("="*50)
    if success:
        print("ALL TESTS PASSED - Try running the full workflow")
    else:
        print("SOME TESTS FAILED - Fix issues before running full workflow")