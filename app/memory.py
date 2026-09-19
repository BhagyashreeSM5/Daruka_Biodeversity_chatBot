from app.db import SessionLocal, ConversationState
from app.models import SiteInput


def load_state(session_id: str) -> SiteInput:
    session = SessionLocal()
    try:
        row = session.get(ConversationState, session_id)
        if row and row.state_json:
            return SiteInput(**row.state_json)
        return SiteInput()
    finally:
        session.close()


def save_state(session_id: str, site_input: SiteInput) -> None:
    session = SessionLocal()
    try:
        row = session.get(ConversationState, session_id)
        payload = site_input.model_dump()
        if row:
            row.state_json = payload
        else:
            row = ConversationState(session_id=session_id, state_json=payload)
            session.add(row)
        session.commit()
    finally:
        session.close()


def clear_state(session_id: str) -> None:
    session = SessionLocal()
    try:
        row = session.get(ConversationState, session_id)
        if row:
            session.delete(row)
            session.commit()
    finally:
        session.close()


def load_turn_count(session_id: str) -> int:
    session = SessionLocal()
    try:
        row = session.get(ConversationState, session_id)
        return row.turn_count if row and row.turn_count else 0
    finally:
        session.close()


def increment_turn_count(session_id: str) -> int:
    """Increment and return the new turn count. Creates the row if needed."""
    session = SessionLocal()
    try:
        row = session.get(ConversationState, session_id)
        if row:
            row.turn_count = (row.turn_count or 0) + 1
            new_count = row.turn_count
        else:
            row = ConversationState(session_id=session_id, state_json={}, turn_count=1)
            session.add(row)
            new_count = 1
        session.commit()
        return new_count
    finally:
        session.close()


def load_and_increment_state(session_id: str) -> tuple[SiteInput, int]:
    """Load conversation state and increment turn count in a single DB query,
    halving network roundtrips to the remote database."""
    session = SessionLocal()
    try:
        row = session.get(ConversationState, session_id)
        if row:
            row.turn_count = (row.turn_count or 0) + 1
            new_count = row.turn_count
            state = SiteInput(**row.state_json) if row.state_json else SiteInput()
        else:
            row = ConversationState(session_id=session_id, state_json={}, turn_count=1)
            session.add(row)
            new_count = 1
            state = SiteInput()
        session.commit()
        return state, new_count
    finally:
        session.close()
