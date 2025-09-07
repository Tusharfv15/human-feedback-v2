"""
Quick setup script for LangSmith tracing configuration.
"""

from setup_langsmith import setup_langsmith_tracing, verify_langsmith_setup

def main():
    """Configure LangSmith tracing for the doping workflow"""
    
    print("🔧 LangSmith Tracing Configuration")
    print("=" * 40)
    
    # Check current status
    if verify_langsmith_setup():
        print("\n✅ LangSmith is already configured!")
        choice = input("\nDo you want to reconfigure? (y/N): ").strip().lower()
        if choice != 'y':
            print("Keeping existing configuration.")
            return
    
    # Get API key from user
    print("\n📝 LangSmith Setup")
    print("Get your API key from: https://smith.langchain.com/")
    api_key = input("\nEnter your LangSmith API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting.")
        return
    
    # Optional: custom project name
    project_name = input("Project name (default: doping-agent-workflow): ").strip()
    if not project_name:
        project_name = "doping-agent-workflow"
    
    # Setup tracing
    print(f"\n🔧 Configuring LangSmith for project: {project_name}")
    env_vars = setup_langsmith_tracing(api_key=api_key, project_name=project_name)
    
    if env_vars:
        print("\n✅ LangSmith tracing configured successfully!")
        print("\n📋 Environment variables set for this session:")
        for key, value in env_vars.items():
            if key == "LANGSMITH_API_KEY":
                print(f"   {key}=***hidden***")
            else:
                print(f"   {key}={value}")
        
        print(f"\n🔗 View traces at: https://smith.langchain.com/o/default/projects/p/{project_name}")
        print("\n💡 To make these settings permanent, add them to your .env file:")
        print("   LANGSMITH_TRACING=true")
        print("   LANGSMITH_ENDPOINT=https://api.smith.langchain.com")
        print(f"   LANGSMITH_API_KEY={api_key}")
        print(f"   LANGSMITH_PROJECT={project_name}")
        
        # Offer to create .env file
        create_env = input("\nCreate/update .env file with these settings? (y/N): ").strip().lower()
        if create_env == 'y':
            try:
                with open('.env', 'a') as f:
                    f.write(f"\n# LangSmith tracing configuration\n")
                    f.write(f"LANGSMITH_TRACING=true\n")
                    f.write(f"LANGSMITH_ENDPOINT=https://api.smith.langchain.com\n")
                    f.write(f"LANGSMITH_API_KEY={api_key}\n")
                    f.write(f"LANGSMITH_PROJECT={project_name}\n")
                print("✅ Settings saved to .env file!")
            except Exception as e:
                print(f"❌ Error creating .env file: {e}")
    else:
        print("❌ Failed to configure LangSmith.")

if __name__ == "__main__":
    main()