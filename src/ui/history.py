"""
Activity History Manager for the AI Visual Assistant.
Provides lightweight session persistence for recent scene analyses,
object detections, OCR readings, and Q&A sessions.
"""

import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import streamlit as st


class ActivityHistory:
    """Manages recent user activity records in session state."""

    SESSION_KEY = "activity_history_records"

    @classmethod
    def get_records(cls) -> List[Dict[str, Any]]:
        """Retrieves history records from session state."""
        if cls.SESSION_KEY not in st.session_state:
            st.session_state[cls.SESSION_KEY] = []
        return st.session_state[cls.SESSION_KEY]

    @classmethod
    def add_record(
        cls,
        task_type: str,
        summary: str,
        details: Optional[Dict[str, Any]] = None,
        image_preview: Optional[Any] = None,
        spoken_text: Optional[str] = None
    ) -> None:
        """Adds a new activity record to history."""
        records = cls.get_records()
        now = datetime.now()
        record = {
            "id": f"act_{int(time.time() * 1000)}",
            "timestamp": now.strftime("%I:%M %p · %b %d, %Y"),
            "task_type": task_type,
            "summary": summary,
            "details": details or {},
            "spoken_text": spoken_text or "",
            "has_image": image_preview is not None,
        }
        # Prepend so newest is first; keep last 20
        records.insert(0, record)
        if len(records) > 20:
            records.pop()
        st.session_state[cls.SESSION_KEY] = records

    @classmethod
    def clear(cls) -> None:
        """Clears all activity history records."""
        st.session_state[cls.SESSION_KEY] = []
