import gradio as gr
import sys
import os

sys.path.append(os.path.dirname(__file__))
from classify import classify_log

def classify_log_ui(source, log_message):
    try:
        label = classify_log(source, log_message)
        return label or "Unclassified"
    except Exception as e:
        return f"Error: {str(e)}"

# Sample examples
examples = [
    ["ModernCRM", "IP 192.168.133.114 blocked due to potential attack"],
    ["BillingSystem", "User 12345 logged in."],
    ["AnalyticsEngine", "Backup completed successfully."],
    ["ModernHR", "GET /v2/servers/detail HTTP/1.1 RCODE 200"],
    ["LegacyCRM", "Case escalation for ticket ID 7324 failed"],
]

# Create Gradio UI
demo = gr.Interface(
    fn=classify_log_ui,
    inputs=[
        gr.Dropdown(
            choices=[
                "ModernCRM",
                "BillingSystem",
                "AnalyticsEngine",
                "ModernHR",
                "LegacyCRM"
            ],
            label="Log Source",
            value="ModernCRM"
        ),
        gr.Textbox(
            label="Log Message",
            placeholder="Enter log message here...",
            lines=3
        )
    ],
    outputs=gr.Textbox(label="Predicted Label"),
    title="🔍 Log Classification System",
    description="""
    ## AI-Powered Log Classification
    Classifies system logs using Regex + BERT + LLM
    
    Categories:
    - Security Alert
    - User Action
    - System Notification
    - HTTP Request
    - Unclassified
    """,
    examples=examples,
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()