import os
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_database_path():
    project_root = get_project_root()
    database_dir = os.path.join(project_root, "database")
    os.makedirs(database_dir, exist_ok=True)

    return os.path.join(database_dir, "chatbot_logs.db")


def get_connection():
    db_path = get_database_path()
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_database():
    """
    Create required database tables if they do not exist.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT NOT NULL,
            bot_reply TEXT NOT NULL,
            intent TEXT,
            confidence REAL,
            source TEXT,
            faq_match_score REAL,
            faq_question TEXT,
            feedback TEXT DEFAULT NULL,
            feedback_comment TEXT DEFAULT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


def save_chat_log(
    user_message: str,
    bot_reply: str,
    intent: Optional[str],
    confidence: Optional[float],
    source: Optional[str],
    faq_match_score: Optional[float] = None,
    faq_question: Optional[str] = None
) -> int:
    """
    Save one chatbot interaction and return log id.
    """

    init_database()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO chat_logs (
            user_message,
            bot_reply,
            intent,
            confidence,
            source,
            faq_match_score,
            faq_question,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_message,
            bot_reply,
            intent,
            confidence,
            source,
            faq_match_score,
            faq_question,
            datetime.utcnow().isoformat()
        )
    )

    log_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return log_id


def save_feedback(
    log_id: int,
    feedback: str,
    feedback_comment: Optional[str] = None
) -> bool:
    """
    Save feedback for an existing chat log.

    feedback should be:
    - helpful
    - not_helpful
    """

    init_database()

    if feedback not in ["helpful", "not_helpful"]:
        raise ValueError("feedback must be either 'helpful' or 'not_helpful'")

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE chat_logs
        SET feedback = ?, feedback_comment = ?
        WHERE id = ?
        """,
        (
            feedback,
            feedback_comment,
            log_id
        )
    )

    updated = cursor.rowcount > 0

    connection.commit()
    connection.close()

    return updated


def get_recent_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get recent chat logs.
    """

    init_database()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM chat_logs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()
    connection.close()

    return [dict(row) for row in rows]


def get_feedback_summary() -> Dict[str, Any]:
    """
    Return feedback and source summary for dashboard/reporting.
    """

    init_database()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("SELECT COUNT(*) AS total FROM chat_logs")
    total = cursor.fetchone()["total"]

    cursor.execute(
        """
        SELECT feedback, COUNT(*) AS count
        FROM chat_logs
        GROUP BY feedback
        """
    )
    feedback_rows = cursor.fetchall()

    cursor.execute(
        """
        SELECT source, COUNT(*) AS count
        FROM chat_logs
        GROUP BY source
        """
    )
    source_rows = cursor.fetchall()

    cursor.execute(
        """
        SELECT intent, COUNT(*) AS count
        FROM chat_logs
        GROUP BY intent
        ORDER BY count DESC
        """
    )
    intent_rows = cursor.fetchall()

    connection.close()

    return {
        "total_logs": total,
        "feedback_counts": [dict(row) for row in feedback_rows],
        "source_counts": [dict(row) for row in source_rows],
        "intent_counts": [dict(row) for row in intent_rows]
    }


def get_logs_by_feedback(feedback: str, limit: int = 100):
    """
    Get logs filtered by feedback.
    feedback can be: helpful, not_helpful
    """

    init_database()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM chat_logs
        WHERE feedback = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (feedback, limit)
    )

    rows = cursor.fetchall()
    connection.close()

    return [dict(row) for row in rows]


def get_unknown_or_low_confidence_logs(limit: int = 100):
    """
    Get logs where chatbot returned unknown or low-confidence response.
    """

    init_database()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM chat_logs
        WHERE intent = 'unknown'
           OR source = 'model_low_confidence'
           OR confidence < 0.60
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()
    connection.close()

    return [dict(row) for row in rows]


def get_all_logs():
    """
    Get all chat logs for export.
    """

    init_database()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM chat_logs
        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()
    connection.close()

    return [dict(row) for row in rows]


def get_dashboard_metrics():
    """
    Get dashboard metrics in a simple format.
    """

    init_database()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("SELECT COUNT(*) AS total FROM chat_logs")
    total_logs = cursor.fetchone()["total"]

    cursor.execute(
        """
        SELECT COUNT(*) AS count
        FROM chat_logs
        WHERE feedback = 'helpful'
        """
    )
    helpful_count = cursor.fetchone()["count"]

    cursor.execute(
        """
        SELECT COUNT(*) AS count
        FROM chat_logs
        WHERE feedback = 'not_helpful'
        """
    )
    not_helpful_count = cursor.fetchone()["count"]

    cursor.execute(
        """
        SELECT COUNT(*) AS count
        FROM chat_logs
        WHERE feedback IS NULL
        """
    )
    no_feedback_count = cursor.fetchone()["count"]

    cursor.execute(
        """
        SELECT COUNT(*) AS count
        FROM chat_logs
        WHERE intent = 'unknown'
           OR source = 'model_low_confidence'
        """
    )
    unknown_count = cursor.fetchone()["count"]

    cursor.execute(
        """
        SELECT AVG(confidence) AS avg_confidence
        FROM chat_logs
        WHERE confidence IS NOT NULL
        """
    )
    avg_confidence = cursor.fetchone()["avg_confidence"]

    connection.close()

    return {
        "total_logs": total_logs,
        "helpful_count": helpful_count,
        "not_helpful_count": not_helpful_count,
        "no_feedback_count": no_feedback_count,
        "unknown_count": unknown_count,
        "avg_confidence": avg_confidence or 0
    }


def get_improvement_candidate_logs(limit: int = 300):
    """
    Get logs that are useful for improving chatbot dataset, FAQ, or rules.

    Includes:
    - not helpful feedback
    - unknown intent
    - low confidence
    - low confidence model fallback
    - FAQ low-confidence cases if stored
    """

    init_database()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM chat_logs
        WHERE feedback = 'not_helpful'
           OR intent = 'unknown'
           OR source = 'model_low_confidence'
           OR confidence < 0.65
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()
    connection.close()

    return [dict(row) for row in rows]