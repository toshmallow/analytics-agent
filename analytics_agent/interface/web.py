"""Web interface for the analytics agent."""

import json
import os
import sys
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, send_from_directory, session, stream_with_context

from analytics_agent.agent.core import AnalyticsAgent
from analytics_agent.container import container

# Get the directory where this file is located
interface_dir = Path(__file__).parent

app = Flask(
    __name__,
    template_folder=str(interface_dir / "templates"),
    static_folder=str(interface_dir / "static"),
)

# Set a secret key for sessions (use environment variable or generate one)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")

# Configure session to persist
app.config["SESSION_PERMANENT"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = 86400  # 24 hours

# In-memory store for agent instances per session
# In production, consider using Redis or a database
_agent_store: dict[str, AnalyticsAgent] = {}


def create_app() -> Flask:
    """Create and configure the Flask application.

    Returns:
        Configured Flask application
    """
    config = container.config()

    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nPlease set up your .env file with required configuration:")
        print("  - GCP_PROJECT_ID")
        print("  - GEMINI_API_KEY")
        print("  - GOOGLE_APPLICATION_CREDENTIALS (optional)")
        sys.exit(1)

    return app


@app.route("/")
def index() -> str:
    """Render the main chat interface.

    Returns:
        HTML page with chat interface
    """
    return render_template("index.html")


@app.route("/api/visualizations/<path:filename>")
def serve_visualization(filename: str) -> Response:
    """Serve visualization images.

    Args:
        filename: Name of the visualization file (can include subdirectories)

    Returns:
        Image file response
    """
    # Get the exports directory (same as visualization tools use)
    exports_dir = Path("exports").resolve()

    # Normalize the filename path
    file_path = Path(filename)

    # Remove "exports" prefix if present in the path
    if len(file_path.parts) > 0 and file_path.parts[0] == "exports":
        file_path = Path(*file_path.parts[1:])

    # Ensure the file is within exports directory for security
    full_path = (exports_dir / file_path).resolve()
    try:
        full_path.relative_to(exports_dir)
    except ValueError:
        # Path is outside exports directory, return 404
        return jsonify({"error": "File not found"}), 404

    # Use send_from_directory with the base directory and relative path
    return send_from_directory(str(exports_dir), str(file_path))


def get_agent() -> AnalyticsAgent:
    """Get or create an agent instance for the current session.

    Returns:
        AnalyticsAgent instance for the current session
    """
    session_id = session.get("session_id")
    if not session_id:
        session_id = os.urandom(16).hex()
        session["session_id"] = session_id
        session.permanent = True  # Make session persist

    if session_id not in _agent_store:
        _agent_store[session_id] = container.analytics_agent()

    return _agent_store[session_id]


@app.route("/api/chat", methods=["POST"])
def chat() -> Response:
    """Handle chat messages and stream responses.

    Returns:
        Streaming response with agent events
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Message is required"}), 400

    # Get agent instance before streaming (capture session context)
    agent = get_agent()
    tools = agent.tools

    def generate() -> Any:
        """Generate streaming response from agent."""
        try:
            # Use the captured agent instance
            for event in agent.analyze(message):
                for node_name, node_output in event.items():
                    if node_name == "agent":
                        messages = node_output.get("messages", [])
                        if messages:
                            last_message = messages[-1]
                            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                                if last_message.content:
                                    yield f"data: {json.dumps({'type': 'agent_thinking', 'content': last_message.content})}\n\n"

                                yield f"data: {json.dumps({'type': 'tool_execution_start'})}\n\n"
                                for tool_call in last_message.tool_calls:
                                    tool_name = tool_call.get("name", "unknown")
                                    tool_args = tool_call.get("args", {})

                                    formatted_call = None
                                    for tool in tools:
                                        if tool_name in tool.get_tool_names():
                                            formatted_call = tool.format_tool_call(
                                                tool_name, tool_args
                                            )
                                            break

                                    if formatted_call:
                                        yield f"data: {json.dumps({'type': 'tool_call', 'content': formatted_call})}\n\n"

                            elif last_message.content:
                                yield f"data: {json.dumps({'type': 'response', 'content': last_message.content})}\n\n"

                    elif node_name == "tools":
                        messages = node_output.get("messages", [])
                        for msg in messages:
                            if hasattr(msg, "content") and hasattr(msg, "name"):
                                tool_name = msg.name
                                result_content = str(msg.content)

                                formatted_result = None
                                for tool in tools:
                                    if tool_name in tool.get_tool_names():
                                        formatted_result = tool.format_tool_result(
                                            tool_name, result_content
                                        )
                                        break

                                if formatted_result:
                                    yield f"data: {json.dumps({'type': 'tool_result', 'content': formatted_result})}\n\n"

                                # Check if this is a visualization result and extract image path
                                if tool_name == "create_visualization" and "Saved to:" in result_content:
                                    # Extract file path from result
                                    try:
                                        file_path = result_content.split("Saved to: ")[1].strip()
                                        # Convert to relative path for web serving
                                        path_obj = Path(file_path)

                                        # Get the exports directory (same as visualization tools use)
                                        exports_dir = Path("exports").resolve()

                                        # Try to make path relative to exports directory
                                        if path_obj.is_absolute():
                                            try:
                                                relative_path = path_obj.relative_to(exports_dir)
                                                filename = str(relative_path).replace("\\", "/")  # Normalize path separators
                                            except ValueError:
                                                # If can't make relative, check if it's in a subdirectory
                                                if "exports" in str(path_obj):
                                                    # Extract the part after exports
                                                    parts = str(path_obj).split("exports")
                                                    if len(parts) > 1:
                                                        filename = parts[1].lstrip("/\\").replace("\\", "/")
                                                    else:
                                                        filename = path_obj.name
                                                else:
                                                    filename = path_obj.name
                                        else:
                                            # Already relative, use as-is but normalize
                                            filename = str(path_obj).replace("\\", "/")

                                        # Send visualization image event
                                        image_url = f"/api/visualizations/{filename}"
                                        yield f"data: {json.dumps({'type': 'visualization', 'url': image_url, 'filename': filename})}\n\n"
                                    except Exception as e:
                                        # If parsing fails, log and continue
                                        print(f"Error parsing visualization path: {e}")
                                        pass

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/reset", methods=["POST"])
def reset() -> Response:
    """Reset the conversation history.

    Returns:
        JSON response confirming reset
    """
    agent = get_agent()
    agent.reset_conversation()
    return jsonify({"status": "success", "message": "Conversation history cleared"})


def main() -> None:
    """Run the web server."""
    import os

    app_instance = create_app()
    port = int(os.getenv("PORT", "8080"))
    print("Starting Analytics Agent web server...")
    print(f"Open your browser and navigate to http://localhost:{port}")
    app_instance.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()

