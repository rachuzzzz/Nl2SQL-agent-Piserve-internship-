"""
Flask backend for the NL2SQL chat UI.

Run:
  pip install flask flask-cors --break-system-packages
  python3 server.py

Then open http://localhost:5000 in your browser.

Set MOCK_MODE = True in nl2sql_ui.html to test without the agent running,
or set it to False to use the real backend.
"""

import json
import os
import sys
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add parent dir to path so we can import the agent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, static_folder='.')
CORS(app)

# ── Load the agent once at startup ─────────────────────────────────────────
print("Loading AgentOrchestrator...")
try:
    from agent.orchestrator import AgentOrchestrator, ConversationSession
    orch = AgentOrchestrator.from_env()
    print("✓ Agent ready")
except Exception as e:
    print(f"✗ Agent failed to load: {e}")
    traceback.print_exc()
    orch = None

# Per-session conversation memory (keyed by session cookie / simple dict for now)
sessions: dict[str, ConversationSession] = {}

# ── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'nl2sql_ui.html')

@app.route('/api/query', methods=['POST'])
def query():
    if orch is None:
        return jsonify({'error': 'Agent not loaded. Check server logs.'}), 503

    data = request.get_json(force=True)
    question = (data.get('question') or '').strip()
    session_id = request.headers.get('X-Session-Id', 'default')

    if not question:
        return jsonify({'error': 'Empty question'}), 400

    # Get or create session
    if session_id not in sessions:
        sessions[session_id] = ConversationSession()
    session = sessions[session_id]

    try:
        result = orch.query(question, session=session)

        # Extract result data from the last execute_sql step
        result_data = None
        last_sql = ''
        for step in reversed(result.steps):
            if step.tool == 'execute_sql' and step.result.get('success'):
                raw = step.result.get('result', {})
                if raw and raw.get('rows'):
                    result_data = {
                        'rows':        raw.get('rows', []),
                        'columns':     raw.get('columns', []),
                        'row_count':   raw.get('row_count', 0),
                        'total_count': raw.get('total_count', 0),
                        'truncated':   raw.get('truncated', False),
                    }
                last_sql = step.args.get('sql', '')
                break

        # Serialise steps (dataclass → dict)
        steps_out = []
        for step in result.steps:
            steps_out.append({
                'iteration':   step.iteration,
                'tool':        step.tool,
                'thought':     step.thought,
                'args':        _safe_json(step.args),
                'result':      _safe_json(step.result),
                'duration_ms': step.duration_ms,
            })

        # Serialise stats
        stats_out = {}
        if result.stats:
            s = result.stats
            stats_out = {
                'total_wall_ms':          s.total_wall_ms,
                'total_llm_calls':        s.total_llm_calls,
                'total_prompt_tokens':    s.total_prompt_tokens,
                'total_completion_tokens':s.total_completion_tokens,
                'total_llm_latency_ms':   s.total_llm_latency_ms,
                'total_sql_exec_ms':      s.total_sql_exec_ms,
            }

        return jsonify({
            'answer':      result.answer,
            'success':     result.success,
            'steps':       steps_out,
            'stats':       stats_out,
            'result_data': result_data,
            'sql':         last_sql,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'answer': f'Server error: {e}'}), 500


@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok' if orch else 'degraded',
        'agent': orch is not None,
    })


def _safe_json(obj):
    """Convert object to JSON-safe dict/list, truncating large values."""
    try:
        raw = json.dumps(obj, default=str)
        if len(raw) > 8000:
            # Truncate large SQL results in steps
            if isinstance(obj, dict) and 'result' in obj:
                obj = dict(obj)
                obj['result'] = {'_truncated': True, 'size': len(raw)}
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return {'_error': 'not serializable'}


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"\n  → Open http://localhost:{port} in your browser\n")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)