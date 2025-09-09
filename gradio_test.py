"""
Simple Gradio test to verify installation and basic functionality
"""

import gradio as gr

def simple_test(text):
    return f"✅ Gradio is working! You entered: {text}"

# Create simple interface
with gr.Blocks(title="Gradio Test") as demo:
    gr.Markdown("# 🧪 Gradio Connection Test")
    
    with gr.Row():
        input_text = gr.Textbox(label="Test Input", placeholder="Enter anything...")
        output_text = gr.Textbox(label="Test Output")
    
    test_btn = gr.Button("Test Connection")
    test_btn.click(simple_test, inputs=input_text, outputs=output_text)

if __name__ == "__main__":
    print("🧪 Starting Gradio connection test...")
    print("📍 Test interface at: http://localhost:7860")
    
    demo.launch(
        server_port=7860,
        debug=True,
        inbrowser=True
    )