"""SQLite investigation records. Each write and its audit event commit together."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3

from fraud_ops.history import behavioural_rules, history_features, parse_time
from fraud_ops.workflow import active_rule_policy

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "investigations.sqlite3"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class ConflictError(ValueError):
    pass


class InvestigationStore:
    def __init__(self, path=None):
        self.path = Path(path or os.environ.get("FRAUD_DB_PATH", DEFAULT_DB))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS transactions (
                    transaction_id TEXT PRIMARY KEY, rail TEXT NOT NULL, profile TEXT NOT NULL,
                    event_time TEXT NOT NULL, received_at TEXT NOT NULL,
                    sender_id TEXT, recipient_id TEXT, transaction_type TEXT,
                    amount REAL NOT NULL, synthetic INTEGER NOT NULL,
                    input_json TEXT NOT NULL, record_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS tx_sender_time ON transactions(rail, sender_id, event_time);
                CREATE INDEX IF NOT EXISTS tx_recipient_time ON transactions(rail, recipient_id, event_time);
                CREATE TABLE IF NOT EXISTS cases (
                    case_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT NOT NULL UNIQUE REFERENCES transactions(transaction_id),
                    status TEXT NOT NULL DEFAULT 'OPEN', assignee TEXT, decision TEXT, reason TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS case_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id INTEGER NOT NULL REFERENCES cases(case_id),
                    event_time TEXT NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL, details_json TEXT NOT NULL
                );
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def ingest(self, transaction, assessment, policy, *, force_case=False):
        tx = dict(transaction)
        tx["event_time"] = parse_time(tx["event_time"]).isoformat()
        if not tx["transaction_id"].strip():
            raise ValueError("Transaction ID is required")
        canonical = json.dumps({"transaction": tx, "force_case": force_case}, sort_keys=True, allow_nan=False)
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT input_json, record_json FROM transactions WHERE transaction_id=?", (tx["transaction_id"],)).fetchone()
            if existing:
                if existing["input_json"] != canonical:
                    raise ConflictError("Transaction ID already exists with different input")
                return {**json.loads(existing["record_json"]), "duplicate": True}
            # Only fetch matching observed entities. Future events cannot affect the snapshot.
            prior = db.execute("""SELECT transaction_id, rail, event_time, sender_id, recipient_id, amount, synthetic
                FROM transactions WHERE rail=? AND synthetic=? AND event_time < ? AND (
                    (sender_id=? AND ? IS NOT NULL) OR (recipient_id=? AND ? IS NOT NULL) OR
                    (recipient_id=? AND ? IS NOT NULL))""",
                (tx["rail"], int(tx.get("synthetic", False)), tx["event_time"], tx.get("sender_id"), tx.get("sender_id"),
                 tx.get("recipient_id"), tx.get("recipient_id"), tx.get("sender_id"), tx.get("sender_id"))).fetchall()
            features = history_features([dict(row) for row in prior], tx)
            rule_policy = active_rule_policy(tx["profile"], assessment["model_version"])
            rules = behavioural_rules(features, tx, rule_policy, assessment["risk_score"])
            decision = policy.decide(assessment["risk_score"], stage=assessment["assessment_stage"], behavioural_alert=bool(rules))
            record = {**tx, "received_at": utc_now(), "assessment": {**assessment, **decision},
                      "behavioural_features": features, "behavioural_rules": rules, "rule_version": rule_policy.version,
                      "rule_policy": rule_policy.to_dict(),
                      "duplicate": False}
            record["assessment"]["decision_reasons"] = (["MODEL_THRESHOLD"] if assessment["risk_score"] >= policy.flag_threshold else []) + [rule["code"] for rule in rules]
            record_json = json.dumps(record, allow_nan=False)
            db.execute("""INSERT INTO transactions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                       (tx["transaction_id"], tx["rail"], tx["profile"], tx["event_time"], record["received_at"],
                        tx.get("sender_id"), tx.get("recipient_id"), tx.get("transaction_type"), tx["amount"],
                        int(tx.get("synthetic", False)), canonical, record_json))
            if force_case or decision["action"] != "PASS":
                stamp = utc_now()
                cursor = db.execute("INSERT INTO cases(transaction_id,created_at,updated_at) VALUES(?,?,?)", (tx["transaction_id"], stamp, stamp))
                self._event(db, cursor.lastrowid, "system", "CREATED", {"rules": rules, "model_version": assessment["model_version"]})
            return record

    @staticmethod
    def _event(db, case_id, actor, action, details):
        db.execute("INSERT INTO case_events(case_id,event_time,actor,action,details_json) VALUES(?,?,?,?,?)",
                   (case_id, utc_now(), actor, action, json.dumps(details)))

    def get_transaction(self, transaction_id):
        with self.connection() as db:
            row = db.execute("SELECT record_json FROM transactions WHERE transaction_id=?", (transaction_id,)).fetchone()
            if row is None:
                raise KeyError("Transaction not found")
            return json.loads(row["record_json"])

    def list_transactions(self, limit=100, sender_id=None, recipient_id=None):
        query, params = "SELECT record_json FROM transactions WHERE 1=1", []
        if sender_id:
            query += " AND sender_id=?"
            params.append(sender_id)
        if recipient_id:
            query += " AND recipient_id=?"
            params.append(recipient_id)
        with self.connection() as db:
            return [json.loads(row[0]) for row in db.execute(query + " ORDER BY event_time DESC LIMIT ?", [*params, min(max(limit, 1), 1000)])]

    def list_cases(self, status=None, assignee=None, synthetic=None):
        query = """SELECT c.*, t.amount,t.rail,t.synthetic,t.event_time,t.sender_id,t.recipient_id,t.record_json
                   FROM cases c JOIN transactions t USING(transaction_id) WHERE 1=1"""
        params = []
        if status:
            query += " AND c.status=?"
            params.append(status)
        if assignee:
            query += " AND c.assignee=?"
            params.append(assignee)
        if synthetic is not None:
            query += " AND t.synthetic=?"
            params.append(int(synthetic))
        query += " ORDER BY c.created_at DESC,c.case_id DESC"
        with self.connection() as db:
            result = []
            for row in db.execute(query, params):
                item = dict(row)
                record = json.loads(item.pop("record_json"))
                item["risk_score"] = record["assessment"]["risk_score"]
                item["model_version"] = record["assessment"]["model_version"]
                item["action"] = record["assessment"]["action"]
                result.append(item)
            return result

    def get_case(self, case_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone()
            if row is None:
                raise KeyError("Case not found")
            result = dict(row)
            result["events"] = [{**dict(event), "details": json.loads(event["details_json"])} for event in
                                db.execute("SELECT * FROM case_events WHERE case_id=? ORDER BY event_id", (case_id,))]
        result["transaction"] = self.get_transaction(result["transaction_id"])
        return result

    def update_case(self, case_id, *, action, actor, expected_version, assignee=None, decision=None, reason=None):
        actor = actor.strip()
        if not actor:
            raise ValueError("Analyst name is required")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone()
            if row is None:
                raise KeyError("Case not found")
            if row["version"] != expected_version:
                raise ConflictError("Case changed; refresh before saving")
            values = dict(row)
            if action == "ASSIGN":
                if row["status"] == "RESOLVED":
                    raise ValueError("Reopen a resolved case before assigning it")
                if not assignee or not assignee.strip():
                    raise ValueError("Assignee is required")
                values.update(assignee=assignee.strip(), status="IN_REVIEW")
            elif action == "RESOLVE":
                if row["status"] == "RESOLVED":
                    raise ConflictError("Case is already resolved")
                if decision not in ["CONFIRMED_FRAUD", "LEGITIMATE", "INCONCLUSIVE"]:
                    raise ValueError("Choose a supported investigation outcome")
                if not reason or not reason.strip():
                    raise ValueError("A reason is required to resolve a case")
                values.update(status="RESOLVED", decision=decision, reason=reason.strip())
            elif action == "REOPEN":
                if row["status"] != "RESOLVED" or not reason or not reason.strip():
                    raise ValueError("Reopening requires a resolved case and a reason")
                values.update(status="OPEN", decision=None, reason=None)
            elif action == "NOTE":
                if not reason or not reason.strip():
                    raise ValueError("A note cannot be empty")
            else:
                raise ValueError("Unknown case action")
            db.execute("UPDATE cases SET status=?,assignee=?,decision=?,reason=?,updated_at=?,version=version+1 WHERE case_id=?",
                (values["status"], values["assignee"], values["decision"], values["reason"], utc_now(), case_id))
            self._event(db, case_id, actor, action, {"assignee": assignee, "decision": decision, "reason": reason,
                                                   "previous_status": row["status"]})
        return self.get_case(case_id)

    def feedback_rows(self, include_synthetic=False):
        rows = self.list_cases(status="RESOLVED", synthetic=None if include_synthetic else False)
        return [{**row, "confirmed_label": int(row["decision"] == "CONFIRMED_FRAUD")}
                for row in rows if row["decision"] in ["CONFIRMED_FRAUD", "LEGITIMATE"]]
