#!/usr/bin/env python3
"""Query the ASC SQLite database and print summary statistics.

Connects to data/asc.db and reports row counts for every table plus
the last 5 audit events, verifying that persistence is working.

Usage:
    python scripts/query_db.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import create_db_engine, get_session_factory
from db.models import (
    AgentRow,
    AuditEventRow,
    DriftMeasurementRow,
    MetricRow,
    ReducibilityRow,
    RunRow,
    SignatureRow,
)


def main() -> None:
    db_path = str(Path(__file__).parent.parent / "data" / "asc.db")
    if not Path(db_path).exists():
        print(f"Database not found at {db_path}")
        print("Run the pipeline with --persist first:")
        print("  python scripts/full_pipeline.py --persist")
        sys.exit(1)

    engine = create_db_engine(db_path)
    Session = get_session_factory(engine)
    session = Session()

    print("=" * 60)
    print("  ASC Database Summary")
    print("=" * 60)

    counts = {
        "Agents": session.query(AgentRow).count(),
        "Runs": session.query(RunRow).count(),
        "Metrics": session.query(MetricRow).count(),
        "Signatures": session.query(SignatureRow).count(),
        "Drift measurements": session.query(DriftMeasurementRow).count(),
        "Reducibility classifications": session.query(ReducibilityRow).count(),
        "Audit events": session.query(AuditEventRow).count(),
    }

    for label, count in counts.items():
        print(f"  {label:30s} {count:>6d}")

    print()
    print("-" * 60)
    print("  Last 5 Audit Events")
    print("-" * 60)

    recent_events = (
        session.query(AuditEventRow)
        .order_by(AuditEventRow.id.desc())
        .limit(5)
        .all()
    )

    if not recent_events:
        print("  (none)")
    else:
        for evt in reversed(recent_events):
            payload_str = ""
            if evt.payload_json:
                try:
                    payload = json.loads(evt.payload_json)
                    payload_str = json.dumps(payload, separators=(",", ":"))
                    if len(payload_str) > 60:
                        payload_str = payload_str[:57] + "..."
                except (json.JSONDecodeError, TypeError):
                    payload_str = evt.payload_json[:60]
            agent_str = evt.agent_id or "-"
            print(
                f"  [{evt.created_at}] {evt.source_component:30s} "
                f"{evt.event_type:25s} agent={agent_str:8s} {payload_str}"
            )

    print()
    session.close()


if __name__ == "__main__":
    main()
