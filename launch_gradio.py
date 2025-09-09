"""
Simple launcher for the Gradio interface
"""

import sys
import subprocess

def install_gradio():
    """Install Gradio if not already installed"""
    try:
        import gradio
        print("✅ Gradio already installed")
        return True
    except ImportError:
        print("📦 Installing Gradio...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio>=4.0.0"])
            print("✅ Gradio installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Gradio: {e}")
            return False

def launch_interface():
    """Launch the Gradio interface"""
    if not install_gradio():
        return
    
    try:
        from gradio_interface import create_gradio_interface
        
        print("🚀 Launching Multi-Agent Semiconductor Processing Interface...")
        print("📍 Interface will be available at: http://127.0.0.1:7860")
        print("⚡ Starting model loading... This may take a moment...")
        
        interface = create_gradio_interface()
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True,
            inbrowser=True  # Automatically open in browser
        )
        
    except Exception as e:
        print(f"❌ Error launching interface: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure all requirements are installed: pip install -r requirements.txt")
        print("2. Check if the model files are accessible")
        print("3. Ensure sufficient GPU memory is available")

if __name__ == "__main__":
    launch_interface()