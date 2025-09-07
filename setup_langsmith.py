"""
Helper script to set up LangSmith tracing for the doping agent workflow
"""

import os
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()

def setup_langsmith_tracing(
    api_key: Optional[str] = None,
    project_name: str = "doping-agent-workflow",
    endpoint: str = "https://api.smith.langchain.com"
) -> Dict[str, str]:
    """
    Set up LangSmith environment variables for tracing
    
    Args:
        api_key: Your LangSmith API key (if None, will try to get from env)
        project_name: Name of the LangSmith project
        endpoint: LangSmith API endpoint
    
    Returns:
        Dict of environment variables set
    """
    
    # Get API key from parameter or environment
    if api_key is None:
        api_key = os.getenv("LANGSMITH_API_KEY")
        if not api_key:
            print("⚠ Warning: No LangSmith API key provided.")
            print("Set LANGSMITH_API_KEY environment variable or pass api_key parameter.")
            print("Get your API key from: https://smith.langchain.com/")
            return {}
    
    # Set environment variables (using correct LangSmith variable names)
    env_vars = {
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_ENDPOINT": endpoint,
        "LANGSMITH_API_KEY": api_key,
        "LANGSMITH_PROJECT": project_name
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"✓ Set {key}={value}")
    
    print(f"\n✅ LangSmith tracing configured for project: {project_name}")
    print(f"🔗 View traces at: https://smith.langchain.com/o/default/projects/p/{project_name}")
    
    return env_vars

def verify_langsmith_setup() -> bool:
    """
    Verify that LangSmith is properly configured
    
    Returns:
        True if setup is valid, False otherwise
    """
    required_vars = [
        "LANGSMITH_TRACING",
        "LANGSMITH_ENDPOINT", 
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ LangSmith setup incomplete. Missing variables:")
        for var in missing_vars:
            print(f"   - {var}")
        return False
    
    print("✅ LangSmith tracing is properly configured")
    print(f"   Project: {os.getenv('LANGSMITH_PROJECT')}")
    print(f"   Endpoint: {os.getenv('LANGSMITH_ENDPOINT')}")
    return True

def disable_langsmith_tracing():
    """Disable LangSmith tracing"""
    os.environ["LANGSMITH_TRACING"] = "false"
    print("🔕 LangSmith tracing disabled")

# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("LANGSMITH SETUP FOR DOPING AGENT WORKFLOW")
    print("=" * 60)
    
    # Check current status
    print("Current LangSmith status:")
    if not verify_langsmith_setup():
        print("\nSetting up LangSmith...")
        
        # Prompt for API key if not set
        api_key = input("Enter your LangSmith API key (or press Enter to skip): ").strip()
        if api_key:
            setup_langsmith_tracing(api_key=api_key)
        else:
            print("Skipping LangSmith setup. You can configure it later.")
            print("Set these environment variables:")
            print("  export LANGSMITH_TRACING=true")
            print("  export LANGSMITH_ENDPOINT=https://api.smith.langchain.com")
            print("  export LANGSMITH_API_KEY=your_api_key_here")
            print("  export LANGSMITH_PROJECT=doping-agent-workflow")
    
    print("\n" + "=" * 60)
    print("Setup complete! You can now run the doping agent workflow with tracing.")