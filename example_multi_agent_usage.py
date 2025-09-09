"""
Example usage of the multi-agent semiconductor processing workflow
"""

from multi_agent_workflow import run_multi_agent_workflow

# Example queries that will be routed to different agents

def example_doping_query():
    """Example query that should route to the doping agent"""
    print("="*60)
    print("EXAMPLE 1: Doping Agent Query")
    print("="*60)
    
    query = "I need to create shallow p+ junctions for a CMOS process with minimal thermal budget"
    result = run_multi_agent_workflow(query)
    return result

def example_etching_query():
    """Example query that should route to the etching agent"""
    print("="*60)
    print("EXAMPLE 2: Etching Agent Query")
    print("="*60)
    
    query = "I need to etch silicon dioxide selectively over silicon nitride using plasma etching"
    result = run_multi_agent_workflow(query)
    return result

def example_wafer_query():
    """Example query that should route to the wafer agent"""
    print("="*60)
    print("EXAMPLE 3: Wafer Agent Query")
    print("="*60)
    
    query = "I need to clean silicon wafers before thermal oxidation to grow 50nm gate oxide"
    result = run_multi_agent_workflow(query)
    return result

def example_multi_agent_query():
    """Example query that might require multiple agents"""
    print("="*60)
    print("EXAMPLE 4: Multi-Agent Query")
    print("="*60)
    
    query = "I need to fabricate CMOS transistors with shallow source/drain junctions, selective gate etching, and proper surface preparation"
    result = run_multi_agent_workflow(query)
    return result

if __name__ == "__main__":
    # Run different examples
    print("Multi-Agent Semiconductor Processing Workflow Examples")
    print("Choose an example to run:")
    print("1. Doping query")
    print("2. Etching query") 
    print("3. Wafer processing query")
    print("4. Multi-agent query")
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        example_doping_query()
    elif choice == "2":
        example_etching_query()
    elif choice == "3":
        example_wafer_query()
    elif choice == "4":
        example_multi_agent_query()
    else:
        print("Invalid choice. Running default doping example...")
        example_doping_query()