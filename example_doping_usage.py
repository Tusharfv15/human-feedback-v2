"""
Example usage of the doping workflow with different query scenarios.
Includes LangSmith tracing setup.
"""

from doping_workflow import run_doping_workflow
from setup_langsmith import setup_langsmith_tracing, verify_langsmith_setup

def main():
    """Run example doping queries"""
    
    print("Doping Workflow Examples with LangSmith Tracing")
    print("=" * 50)
    
    # Setup LangSmith tracing (optional)
    print("Setting up LangSmith tracing...")
    if not verify_langsmith_setup():
        print("⚠ LangSmith not configured. Workflow will run without tracing.")
        print("To enable tracing, set your LANGSMITH_API_KEY environment variable.")
    else:
        print("✅ LangSmith tracing enabled!")
    
    print("\n" + "=" * 50)
    
    # Example queries
    examples = [
        "I need to create shallow p+ junctions for a CMOS process with minimal thermal budget",
        "What parameters should I use for phosphorus implantation to achieve 1e19 carrier concentration at 100nm depth?",
        "I want to optimize arsenic implantation for source/drain regions with low resistance",
    ]
    
    print("Available example queries:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    
    print("\nChoose an example (1-3) or enter your own query:")
    user_input = input("> ").strip()
    
    if user_input.isdigit() and 1 <= int(user_input) <= 3:
        query = examples[int(user_input) - 1]
    else:
        query = user_input
    
    if not query:
        print("No query provided. Exiting.")
        return
    
    print(f"\nRunning workflow with query: {query}")
    print("-" * 60)
    
    try:
        result = run_doping_workflow(query)
        print("\nWorkflow completed successfully!")
        
        # Show LangSmith trace link if available
        if verify_langsmith_setup():
            project_name = "doping-agent-workflow"
            print(f"\n🔗 View detailed traces at: https://smith.langchain.com/o/default/projects/p/{project_name}")
        
    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user.")
    except Exception as e:
        print(f"\nError running workflow: {e}")


if __name__ == "__main__":
    main()