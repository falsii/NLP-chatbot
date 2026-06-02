import os
import sys

import pandas as pd
import streamlit as st


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)


from database import (
    get_recent_logs,
    get_feedback_summary,
    get_logs_by_feedback,
    get_unknown_or_low_confidence_logs,
    get_all_logs,
    get_dashboard_metrics,
    init_database
)


st.set_page_config(
    page_title="Chatbot Admin Dashboard",
    page_icon="📊",
    layout="wide"
)


def dataframe_from_logs(logs):
    if not logs:
        return pd.DataFrame()

    return pd.DataFrame(logs)


def render_metric_cards(metrics):
    total_logs = metrics["total_logs"]
    helpful_count = metrics["helpful_count"]
    not_helpful_count = metrics["not_helpful_count"]
    no_feedback_count = metrics["no_feedback_count"]
    unknown_count = metrics["unknown_count"]
    avg_confidence = metrics["avg_confidence"]

    feedback_total = helpful_count + not_helpful_count

    helpful_rate = 0
    if feedback_total > 0:
        helpful_rate = helpful_count / feedback_total * 100

    unknown_rate = 0
    if total_logs > 0:
        unknown_rate = unknown_count / total_logs * 100

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Total Logs", total_logs)
    col2.metric("Helpful", helpful_count)
    col3.metric("Not Helpful", not_helpful_count)
    col4.metric("No Feedback", no_feedback_count)
    col5.metric("Helpful Rate", f"{helpful_rate:.1f}%")
    col6.metric("Unknown Rate", f"{unknown_rate:.1f}%")

    st.metric("Average Confidence", f"{avg_confidence:.2f}")


def render_summary_charts(summary):
    source_counts = summary.get("source_counts", [])
    intent_counts = summary.get("intent_counts", [])
    feedback_counts = summary.get("feedback_counts", [])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Source Breakdown")
        source_df = pd.DataFrame(source_counts)

        if not source_df.empty:
            source_df = source_df.rename(columns={"source": "Source", "count": "Count"})
            st.bar_chart(source_df.set_index("Source"))
        else:
            st.info("No source data yet.")

    with col2:
        st.subheader("Intent Breakdown")
        intent_df = pd.DataFrame(intent_counts)

        if not intent_df.empty:
            intent_df = intent_df.rename(columns={"intent": "Intent", "count": "Count"})
            st.bar_chart(intent_df.set_index("Intent"))
        else:
            st.info("No intent data yet.")

    with col3:
        st.subheader("Feedback Breakdown")
        feedback_df = pd.DataFrame(feedback_counts)

        if not feedback_df.empty:
            feedback_df["feedback"] = feedback_df["feedback"].fillna("no_feedback")
            feedback_df = feedback_df.rename(columns={"feedback": "Feedback", "count": "Count"})
            st.bar_chart(feedback_df.set_index("Feedback"))
        else:
            st.info("No feedback data yet.")


def render_recent_logs():
    st.subheader("Recent Chat Logs")

    limit = st.slider("Number of recent logs", min_value=10, max_value=200, value=50, step=10)

    logs = get_recent_logs(limit=limit)
    df = dataframe_from_logs(logs)

    if df.empty:
        st.info("No chat logs found yet.")
        return

    columns_to_show = [
        "id",
        "user_message",
        "bot_reply",
        "intent",
        "confidence",
        "source",
        "faq_match_score",
        "feedback",
        "created_at"
    ]

    available_columns = [column for column in columns_to_show if column in df.columns]

    st.dataframe(
        df[available_columns],
        use_container_width=True,
        hide_index=True
    )


def render_not_helpful_logs():
    st.subheader("Not Helpful Responses")

    logs = get_logs_by_feedback("not_helpful", limit=100)
    df = dataframe_from_logs(logs)

    if df.empty:
        st.success("No not-helpful feedback yet.")
        return

    columns_to_show = [
        "id",
        "user_message",
        "bot_reply",
        "intent",
        "confidence",
        "source",
        "feedback_comment",
        "created_at"
    ]

    available_columns = [column for column in columns_to_show if column in df.columns]

    st.dataframe(
        df[available_columns],
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "Use these messages to improve intents.json, faq.json, rules.py, or confidence threshold."
    )


def render_unknown_logs():
    st.subheader("Unknown / Low Confidence Messages")

    logs = get_unknown_or_low_confidence_logs(limit=100)
    df = dataframe_from_logs(logs)

    if df.empty:
        st.success("No unknown or low-confidence messages found.")
        return

    columns_to_show = [
        "id",
        "user_message",
        "bot_reply",
        "intent",
        "confidence",
        "source",
        "created_at"
    ]

    available_columns = [column for column in columns_to_show if column in df.columns]

    st.dataframe(
        df[available_columns],
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "These messages are useful for creating new training patterns or new FAQ entries."
    )


def render_export_section():
    st.subheader("Export Logs")

    logs = get_all_logs()
    df = dataframe_from_logs(logs)

    if df.empty:
        st.info("No logs available to export.")
        return

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Chat Logs CSV",
        data=csv_data,
        file_name="chatbot_logs.csv",
        mime="text/csv"
    )


def render_improvement_notes():
    st.subheader("How to Improve the Chatbot Using This Dashboard")

    st.markdown(
        """
        Use this dashboard like a feedback loop:

        1. Check **Not Helpful Responses**
        2. Check **Unknown / Low Confidence Messages**
        3. Add missing examples to `data/intents.json`
        4. Add missing FAQ answers to `data/faq.json`
        5. Improve overly broad rules in `src/rules.py`
        6. Retrain the model using `python src/train.py`
        7. Run evaluation using `python src/evaluate.py`
        8. Test again in Streamlit and FastAPI
        """
    )


def render_improvement_candidates():
    st.subheader("Improvement Candidates")

    candidates_path = os.path.join(PROJECT_ROOT, "data", "improvement_candidates.json")

    if not os.path.exists(candidates_path):
        st.info("No improvement report found yet. Run: python scripts/generate_improvement_report.py")
        return

    with open(candidates_path, "r", encoding="utf-8") as file:
        report = st.json.load(file)

    summary = report.get("summary", {})
    candidates = report.get("candidates", [])
    repeated_messages = report.get("repeated_problem_messages", [])

    st.markdown("### Summary")
    st.json(summary)

    st.markdown("### Repeated Problem Messages")

    if repeated_messages:
        st.dataframe(pd.DataFrame(repeated_messages), use_container_width=True, hide_index=True)
    else:
        st.info("No repeated problem messages found.")

    st.markdown("### Candidate Logs")

    if candidates:
        df = pd.DataFrame(candidates)
        columns = [
            "log_id",
            "user_message",
            "predicted_intent",
            "confidence",
            "source",
            "feedback",
            "suggested_improvement_type",
            "recommended_action",
            "review_status"
        ]

        available_columns = [column for column in columns if column in df.columns]

        st.dataframe(df[available_columns], use_container_width=True, hide_index=True)
    else:
        st.success("No improvement candidates found.")


def main():
    init_database()

    st.title("Chatbot Admin Dashboard")
    st.caption("Monitor logs, feedback, confidence, sources, and failed predictions.")

    metrics = get_dashboard_metrics()
    summary = get_feedback_summary()

    render_metric_cards(metrics)

    st.divider()

    render_summary_charts(summary)

    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Recent Logs",
            "Not Helpful",
            "Unknown / Low Confidence",
            "Improvement Candidates",
            "Export",
            "Improvement Notes"
        ]
    )

    with tab1:
        render_recent_logs()

    with tab2:
        render_not_helpful_logs()

    with tab3:
        render_unknown_logs()

    with tab4:
        render_export_section()

    with tab5:
        render_improvement_notes()

    with tab6:   
        render_improvement_candidates()


if __name__ == "__main__":
    main()