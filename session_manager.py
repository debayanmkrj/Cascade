"""Session Manager - Track and log all generation sessions (Spec Section 8.1)"""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config import SESSIONS_DIR
from data_types import SessionState, RawInputBundle, NodeBrief


class SessionManager:
    """
    Manages session state for DesignAssistant.
    Logs all events and state changes for long-trail context.
    """

    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}
        self.sessions_dir = Path(SESSIONS_DIR)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, input_bundle: RawInputBundle) -> SessionState:
        """Create a new session"""
        session_id = str(uuid.uuid4())[:8]
        session = SessionState(
            session_id=session_id,
            seed=input_bundle.seed,
            input_bundle=input_bundle,
            node_brief=None,
            history=[],
            phase1_meta={}
        )
        self.sessions[session_id] = session
        self._log_event(session_id, 'session_created', {'seed': input_bundle.seed})
        return session

    def update_brief(self, session_id: str, node_brief: NodeBrief):
        """Update session with generated NodeBrief"""
        if session_id in self.sessions:
            self.sessions[session_id].node_brief = node_brief
            self.sessions[session_id].updated_at = datetime.now()
            self._log_event(session_id, 'brief_generated', {
                'node_count': node_brief.node_count,
                'ui_count': len(node_brief.ui_controls)
            })

    def update_meta(self, session_id: str, meta_key: str, meta_value):
        """Update phase1_meta dict"""
        if session_id in self.sessions:
            self.sessions[session_id].phase1_meta[meta_key] = meta_value
            self.sessions[session_id].updated_at = datetime.now()

    def _log_event(self, session_id: str, event_type: str, data: Dict):
        """Log event to session history"""
        if session_id in self.sessions:
            event = {
                'type': event_type,
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            self.sessions[session_id].history.append(event)

    def log_edit(self, session_id: str, instruction: str, changes: Dict):
        """Log user edit request"""
        self._log_event(session_id, 'user_edit', {
            'instruction': instruction,
            'changes': changes
        })

    def log_reiteration(self, session_id: str, new_seed: int, reason: str):
        """Log re-iteration event"""
        self._log_event(session_id, 'reiteration', {
            'new_seed': new_seed,
            'reason': reason
        })

    def save_session(self, session_id: str) -> Path:
        """Save session to JSON file"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.sessions[session_id]
        filename = self.sessions_dir / f"session_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"✓ Session saved to {filename}")
        return filename

    def load_session(self, session_id: str) -> Optional[SessionState]:
        """Load most recent session file for given ID"""
        pattern = f"session_{session_id}_*.json"
        files = list(self.sessions_dir.glob(pattern))
        if not files:
            return None

        # Load most recent
        latest_file = max(files, key=lambda p: p.stat().st_mtime)
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Reconstruct SessionState (simplified - would need full deserialization)
        print(f"✓ Loaded session from {latest_file}")
        return data  # Return raw dict for now

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get active session by ID"""
        return self.sessions.get(session_id)

    def get_context_for_llm(self, session_id: str, max_events: int = 20) -> str:
        """
        Get session context formatted for LLM.
        Returns recent history as formatted text.
        """
        if session_id not in self.sessions:
            return ""

        session = self.sessions[session_id]
        context_parts = []

        # Add session metadata
        context_parts.append(f"SESSION ID: {session_id}")
        context_parts.append(f"SEED: {session.seed}")
        context_parts.append(f"PROMPT: {session.input_bundle.prompt_text}")
        context_parts.append(f"MODE: {session.input_bundle.user_mode}, D_GLOBAL: {session.input_bundle.D_global}")
        context_parts.append("")

        # Add brief summary if exists
        if session.node_brief:
            brief = session.node_brief
            context_parts.append("CURRENT BRIEF:")
            context_parts.append(f"  Essence: {brief.essence}")
            context_parts.append(f"  UI Controls: {len(brief.ui_controls)}")
            context_parts.append(f"  Node Archetypes: {brief.node_count}")
            context_parts.append(f"  Divergence: S={brief.divergence.D_surface:.2f}, F={brief.divergence.D_flow:.2f}, N={brief.divergence.D_narrative:.2f}")
            context_parts.append("")

        # Add recent history
        context_parts.append("RECENT HISTORY:")
        recent_events = session.history[-max_events:]
        for event in recent_events:
            context_parts.append(f"  [{event['timestamp']}] {event['type']}: {event.get('data', {})}")

        return "\n".join(context_parts)
