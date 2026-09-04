from __future__ import annotations

import re
from datetime import UTC, datetime

from sqlalchemy import delete, or_, select
from sqlalchemy.orm import sessionmaker

from stock_content.adapters.postgres.models import (
    KnowledgeCrossVideoRow,
    KnowledgeEvidenceRow,
    KnowledgeUnitRow,
    LifecycleEventLedgerRow,
)
from stock_content.domain.cross_video_corroboration import CrossVideoCorroborationService
from stock_content.domain.knowledge_enums import support_rank
from stock_content.domain.models import KnowledgeUnit
from stock_content.domain.signal_contract import upgrade_signal_v3


class PostgresKnowledgeRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._sessions = session_factory

    @staticmethod
    def _payload(row: KnowledgeUnitRow) -> dict:
        return {
            "knowledge_uid": row.knowledge_uid,
            "video_id": row.video_id,
            "chapter_id": row.chapter_id,
            "statement": row.statement,
            "kind": row.kind,
            "knowledge_kind": row.knowledge_kind,
            "knowledge_version": row.knowledge_version,
            "subject": row.subject,
            "subject_key": row.subject_key,
            "predicate_key": row.predicate_key,
            "ticker": row.ticker,
            "sentiment": row.sentiment,
            "support_status": row.support_status,
            "truth_status": row.truth_status,
            "review_status": row.review_status,
            "lifecycle_status": row.lifecycle_status,
            "confidence": row.confidence,
            "as_of": row.as_of.isoformat(),
            "as_of_time": row.as_of.isoformat(),
            "available_from": row.available_from.isoformat(),
            # Preserve the complete temporal/occurrence projection.  Search
            # callers use PostgreSQL's hydrated payload as the PIT authority.
            "attributes": dict(row.attributes or {}),
            "valid_from": row.valid_from.isoformat() if row.valid_from else None,
            "valid_to": row.valid_to.isoformat() if row.valid_to else None,
            "provenance": dict(row.provenance or {}),
        }

    @staticmethod
    def _apply_filters(query, filters: dict):
        for field in ("kind", "ticker", "subject", "support_status", "review_status", "video_id"):
            value = filters.get(field)
            if value:
                query = query.where(getattr(KnowledgeUnitRow, field) == value)
        return query

    @staticmethod
    def _pit_matches(payload: dict, filters: dict) -> bool:
        """Apply PIT and temporal predicates to an authoritative SQL payload."""
        mode = str(filters.get("pit_mode") or filters.get("availability_mode") or "SYSTEM").upper()
        if mode not in {"SYSTEM", "PUBLIC_STRICT", "PUBLIC_ALLOW_PROXY"}:
            raise ValueError(f"unknown pit mode: {mode}")
        as_of = filters.get("availability_as_of")
        attributes = dict(payload.get("attributes") or {})
        quality = str(attributes.get("source_availability_quality") or "UNKNOWN").upper()
        if mode == "PUBLIC_STRICT" and (quality != "EXACT" or not attributes.get("source_available_at")):
            return False
        if mode == "PUBLIC_ALLOW_PROXY" and quality not in {
            "EXACT", "PUBLISHED_TIME_PROXY", "INGEST_TIME_UPPER_BOUND"
        }:
            return False
        if as_of is not None:
            as_of_text = _iso(as_of)
            if _iso(payload.get("available_from")) > as_of_text:
                return False
            source_available = attributes.get("source_available_at")
            if mode == "PUBLIC_STRICT":
                if not source_available or _iso(source_available) > as_of_text:
                    return False
            elif mode == "PUBLIC_ALLOW_PROXY":
                if not source_available or _iso(source_available) > as_of_text:
                    return False

        bindings = list(attributes.get("temporal_bindings") or [])
        role = filters.get("temporal_role")
        target_start = filters.get("target_start")
        target_end = filters.get("target_end")
        if role or target_start or target_end:
            candidates = [item for item in bindings if not role or str(item.get("role") or "") == str(role)]
            if target_start or target_end:
                candidates = [item for item in candidates if _binding_overlaps(item, target_start, target_end)]
            if not candidates:
                return False
        segment_id = filters.get("semantic_segment_id")
        if segment_id and str(attributes.get("semantic_segment_id") or "") != str(segment_id):
            return False

        business_as_of = filters.get("business_as_of")
        knowledge_as_of = filters.get("knowledge_as_of")
        if not filters.get("historical_projection") and (business_as_of is not None or knowledge_as_of is not None):
            business = _iso(business_as_of) if business_as_of else "9999-12-31T23:59:59+00:00"
            knowledge = _iso(knowledge_as_of) if knowledge_as_of else "9999-12-31T23:59:59+00:00"
            events = list(attributes.get("lifecycle_events") or [])
            if events:
                visible = [
                    item for item in events
                    if _iso(item.get("effective_at")) <= business
                    and _iso(item.get("recorded_at")) <= knowledge
                ]
                if not visible:
                    return False
                state = max(visible, key=lambda item: (
                    _iso(item.get("effective_at")), _iso(item.get("recorded_at")),
                    str(item.get("lifecycle_event_id") or ""),
                )).get("to_status")
                if state not in {None, "ACTIVE"}:
                    return False
            elif payload.get("lifecycle_status") not in {None, "ACTIVE"}:
                return False
        return True

    @staticmethod
    def _lifecycle_matches(session, payload: dict, filters: dict) -> bool:
        """Resolve bitemporal lifecycle state from the authoritative ledger."""
        business = filters.get("business_as_of") or datetime.max.replace(tzinfo=UTC)
        knowledge = filters.get("knowledge_as_of") or datetime.max.replace(tzinfo=UTC)
        attrs = dict(payload.get("attributes") or {})
        target_specs = []
        if attrs.get("claim_id"):
            target_specs.append(("CLAIM", str(attrs["claim_id"])))
        if attrs.get("occurrence_id"):
            target_specs.append(("OCCURRENCE", str(attrs["occurrence_id"])))
        if not target_specs:
            target_specs.append(("CLAIM", str(payload.get("knowledge_uid") or "")))
        for target_type, target_id in target_specs:
            rows = session.scalars(
                select(LifecycleEventLedgerRow).where(
                    LifecycleEventLedgerRow.target_type == target_type,
                    LifecycleEventLedgerRow.target_id == target_id,
                    LifecycleEventLedgerRow.effective_at <= business,
                    LifecycleEventLedgerRow.recorded_at <= knowledge,
                )
            ).all()
            if not rows:
                if (
                    attrs.get("claim_id")
                    or attrs.get("occurrence_id")
                    or filters.get("business_as_of") is not None
                    or filters.get("knowledge_as_of") is not None
                ):
                    return False
                if payload.get("lifecycle_status") != "ACTIVE":
                    return False
                continue
            latest = max(rows, key=lambda row: (row.effective_at, row.recorded_at, row.lifecycle_event_id))
            if latest.to_status != "ACTIVE":
                return False
        return True

    def replace_for_video(self, video_id: str, units: list[KnowledgeUnit]) -> None:
        with self._sessions.begin() as session:
            existing = {
                row.knowledge_uid: row
                for row in session.scalars(select(KnowledgeUnitRow).where(KnowledgeUnitRow.video_id == video_id))
            }
            incoming = {unit.knowledge_uid for unit in units}
            for knowledge_uid, row in existing.items():
                if knowledge_uid not in incoming:
                    row.lifecycle_status = "SUPERSEDED"
            for unit in units:
                values = vars(unit)
                row = existing.get(unit.knowledge_uid)
                if row is None:
                    row = KnowledgeUnitRow(**values)
                    session.add(row)
                else:
                    for name, value in values.items():
                        setattr(row, name, value)
                session.execute(
                    delete(KnowledgeEvidenceRow).where(KnowledgeEvidenceRow.knowledge_uid == unit.knowledge_uid)
                )
                evidence_items = list((unit.attributes or {}).get("evidence") or [])
                for evidence in evidence_items:
                    session.add(
                        KnowledgeEvidenceRow(
                            knowledge_uid=unit.knowledge_uid,
                            evidence_type=str(
                                evidence.get("evidence_type") or evidence.get("source_type") or "TRANSCRIPT"
                            ),
                            source_id=str(evidence.get("source_id") or "") or None,
                            video_id=video_id,
                            frame_id=str(evidence.get("frame_id") or "") or None,
                            evidence_text=str(evidence.get("text") or evidence.get("evidence_text") or ""),
                            start_seconds=evidence.get("start_seconds"),
                            end_seconds=evidence.get("end_seconds"),
                            structured_payload=dict(evidence.get("structured_payload") or {}),
                            confidence=evidence.get("confidence") or evidence.get("confidence_score"),
                            source_reliability=evidence.get("source_reliability"),
                        )
                    )
            self._refresh_cross_video(session, units)

    @staticmethod
    def _refresh_cross_video(session, units: list[KnowledgeUnit]) -> None:
        """Persist corroboration once at write time; read APIs never invent it."""
        affected = {
            (unit.subject_key or unit.ticker or "", unit.predicate_key or unit.knowledge_kind) for unit in units
        }
        rows = session.scalars(select(KnowledgeUnitRow).where(KnowledgeUnitRow.lifecycle_status == "ACTIVE")).all()
        payload = [
            {
                "knowledge_uid": row.knowledge_uid,
                "video_id": row.video_id,
                "subject_key": row.subject_key,
                "predicate_key": row.predicate_key,
                "sentiment": row.sentiment,
                "support_status": row.support_status,
                "lifecycle_status": row.lifecycle_status,
                "evidence_ids": [
                    str(value)
                    for value in session.scalars(
                        select(KnowledgeEvidenceRow.id).where(KnowledgeEvidenceRow.knowledge_uid == row.knowledge_uid)
                    ).all()
                ],
            }
            for row in rows
            if (row.subject_key or row.ticker or "", row.predicate_key or row.knowledge_kind) in affected
        ]
        for knowledge_uid, values in CrossVideoCorroborationService().calculate(payload).items():
            state = session.get(KnowledgeCrossVideoRow, knowledge_uid)
            # Keep the legacy column as an alias until all clients move to the
            # unambiguous content_attention_score contract.
            values["author_attention_score"] = values["content_attention_score"]
            if state is None:
                session.add(KnowledgeCrossVideoRow(knowledge_uid=knowledge_uid, **values))
            else:
                for name, value in values.items():
                    setattr(state, name, value)

    def get(self, knowledge_uid: str) -> dict | None:
        with self._sessions() as session:
            row = session.get(KnowledgeUnitRow, knowledge_uid)
            return self._payload(row) if row else None

    def list_for_video(self, video_id: str, limit: int) -> list[dict]:
        with self._sessions() as session:
            rows = session.scalars(
                select(KnowledgeUnitRow)
                .where(KnowledgeUnitRow.video_id == video_id)
                .order_by(KnowledgeUnitRow.available_from, KnowledgeUnitRow.knowledge_uid)
                .limit(limit)
            ).all()
            return [self._payload(row) for row in rows]

    def list_for_video_as_of(
        self,
        video_id: str,
        as_of: datetime,
        limit: int = 100,
        *,
        availability_mode: str = "SYSTEM",
        temporal_role: str | None = None,
        semantic_segment_id: str | None = None,
    ) -> list[dict]:
        """PIT read helper; never reads rows newer than the requested clock."""
        with self._sessions() as session:
            rows = session.scalars(
                select(KnowledgeUnitRow)
                .where(KnowledgeUnitRow.video_id == video_id, KnowledgeUnitRow.available_from <= as_of)
                .order_by(KnowledgeUnitRow.available_from, KnowledgeUnitRow.knowledge_uid)
            ).all()
            payloads = [self._payload(row) for row in rows]
        mode = str(availability_mode).upper()
        if mode not in {"SYSTEM", "PUBLIC_STRICT", "PUBLIC_ALLOW_PROXY"}:
            raise ValueError(f"unknown availability mode: {availability_mode}")
        result = []
        for payload in payloads:
            if not self._pit_matches(payload, {
                "availability_as_of": as_of,
                "pit_mode": mode,
                "temporal_role": temporal_role,
                "semantic_segment_id": semantic_segment_id,
            }):
                continue
            quality = str((payload.get("attributes") or {}).get("source_availability_quality") or "UNKNOWN").upper()
            payload["source_availability_quality"] = quality
            result.append(payload)
        # PUBLIC/semantic predicates are evaluated against the authoritative
        # payload after the SQL candidate scan.  Applying ``limit`` in SQL
        # first can hide a later eligible row behind an earlier rejected one.
        return result[:limit]

    list_pit = list_for_video_as_of

    def search(self, query: str, filters: dict, limit: int) -> list[dict]:
        with self._sessions() as session:
            statement = select(KnowledgeUnitRow).where(
                or_(
                    KnowledgeUnitRow.statement.ilike(f"%{query}%"),
                    KnowledgeUnitRow.subject.ilike(f"%{query}%"),
                    KnowledgeUnitRow.ticker.ilike(f"%{query}%"),
                )
            )
            statement = self._apply_filters(statement, filters)
            # Apply the cheap relational cutoff before limiting.  When any
            # post-filter is active, fetch the complete SQL candidate set so
            # rejected high-confidence rows cannot hide later eligible rows.
            cutoff = filters.get("availability_as_of")
            if cutoff is not None:
                statement = statement.where(KnowledgeUnitRow.available_from <= cutoff)
            post_filter = any(
                key in filters for key in (
                    "pit_mode", "availability_mode", "availability_as_of", "temporal_role",
                    "target_start", "target_end", "semantic_segment_id", "business_as_of", "knowledge_as_of",
                )
            )
            query = statement.order_by(KnowledgeUnitRow.confidence.desc())
            rows = session.scalars(query if post_filter else query.limit(limit)).all()
            filtered = [
                payload for row in rows
                if self._pit_matches(payload := self._payload(row), filters)
                and self._lifecycle_matches(session, payload, filters)
            ]
            return filtered[:limit]

    def hydrate(self, knowledge_uids: list[str], filters: dict) -> list[dict]:
        if not knowledge_uids:
            return []
        with self._sessions() as session:
            statement = select(KnowledgeUnitRow).where(KnowledgeUnitRow.knowledge_uid.in_(knowledge_uids))
            statement = self._apply_filters(statement, filters)
            rows = session.scalars(statement).all()
            by_uid = {row.knowledge_uid: self._payload(row) for row in rows}
            return [
                by_uid[uid] for uid in knowledge_uids
                if uid in by_uid
                and self._pit_matches(by_uid[uid], filters)
                and self._lifecycle_matches(session, by_uid[uid], filters)
            ]

    def factor_signals(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        minimum_support_status: str,
    ) -> list[dict]:
        with self._sessions() as session:
            statement = select(KnowledgeUnitRow).where(
                KnowledgeUnitRow.available_from >= start,
                KnowledgeUnitRow.available_from <= end,
                KnowledgeUnitRow.review_status != "REJECTED",
                KnowledgeUnitRow.lifecycle_status == "ACTIVE",
            )
            rows = session.scalars(statement.order_by(KnowledgeUnitRow.available_from)).all()
            minimum = support_rank(minimum_support_status)
            requested_codes = {_ticker_code(symbol) for symbol in symbols if _ticker_code(symbol)}
            eligible = [
                row
                for row in rows
                if support_rank(row.support_status) >= minimum
                and (not requested_codes or _ticker_code(row.ticker or row.subject_key or "") in requested_codes)
            ]
            items = []
            for row in eligible:
                cross = session.get(KnowledgeCrossVideoRow, row.knowledge_uid)
                evidence_ids = [
                    str(value)
                    for value in session.scalars(
                        select(KnowledgeEvidenceRow.id).where(KnowledgeEvidenceRow.knowledge_uid == row.knowledge_uid)
                    ).all()
                ]
                # 收尾文档 §63：Quant market_snapshot_id / data_version / available_at
                # 沿 Evidence/Signal 链路传递（来自 external fact verification 的 PIT 行情锚点）。
                attributes = row.attributes or {}
                market_fact = (attributes.get("external_verification") or {}).get("market_fact") or {}
                items.append(
                    {
                        "signal_id": row.knowledge_uid,
                        "knowledge_uid": row.knowledge_uid,
                        "knowledge_version": row.knowledge_version,
                        "symbol": row.subject_key or row.ticker,
                        "subject": row.subject,
                        "subject_key": row.subject_key or row.ticker or row.subject,
                        "kind": row.kind,
                        "knowledge_kind": row.knowledge_kind,
                        "event_type": (row.attributes or {}).get("event_type"),
                        "sentiment": row.sentiment,
                        "confidence": row.confidence,
                        "support_status": row.support_status,
                        "truth_status": row.truth_status,
                        "review_status": row.review_status,
                        "lifecycle_status": row.lifecycle_status,
                        "as_of_time": row.as_of.isoformat(),
                        "available_from": row.available_from.isoformat(),
                        "content_attention_score": cross.content_attention_score if cross else 0.0,
                        "author_attention_score": cross.content_attention_score if cross else 0.0,
                        "event_strength": (row.attributes or {}).get("event_strength", row.confidence),
                        "cross_video_consensus": cross.consensus_score if cross else 0.0,
                        "cross_video_disagreement": cross.disagreement_score if cross else 0.0,
                        "source_video_id": row.video_id,
                        "evidence_ids": evidence_ids,
                        "external_verification_status": attributes.get("external_verification_status"),
                        "market_snapshot_id": market_fact.get("data_snapshot_id"),
                        "market_data_version": market_fact.get("data_version"),
                        "market_fact_date": market_fact.get("trading_date"),
                        "content_snapshot_id": attributes.get("content_snapshot_id"),
                        "claim_id": attributes.get("claim_id"),
                        "provenance": dict(row.provenance or {}),
                    }
                )
            # §7：content-factor-signal.v3 —— 正式版本化的金融研究输入。
            return [upgrade_signal_v3(item) for item in items]

    def factor_signals_v5(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        minimum_support_status: str | int,
        *,
        availability_as_of: datetime | None = None,
        pit_mode: str | None = None,
        business_as_of: datetime | None = None,
        knowledge_as_of: datetime | None = None,
        historical_projection: bool = False,
        claim_ids: list[str] | tuple[str, ...] | None = None,
    ) -> list[dict]:
        """Build lineage-only v5 signals from SQL rows after PIT filtering."""
        from stock_content.domain.signal_contract import upgrade_signal_v5

        with self._sessions() as session:
            rows = session.scalars(
                select(KnowledgeUnitRow)
                .where(
                    KnowledgeUnitRow.available_from >= start,
                    KnowledgeUnitRow.available_from <= end,
                    KnowledgeUnitRow.review_status != "REJECTED",
                )
                .order_by(KnowledgeUnitRow.available_from, KnowledgeUnitRow.knowledge_uid)
            ).all()
            minimum = (
                minimum_support_status
                if isinstance(minimum_support_status, int) and not isinstance(minimum_support_status, bool)
                else support_rank(minimum_support_status)
            )
            requested = {_ticker_code(symbol) for symbol in symbols if _ticker_code(symbol)}
            requested_claims = {str(item) for item in (claim_ids or ())}
            items: list[dict] = []
            for row in rows:
                payload = self._payload(row)
                if requested_claims and str(payload.get("claim_id") or "") not in requested_claims:
                    continue
                if support_rank(row.support_status) < minimum:
                    continue
                if requested and _ticker_code(row.ticker or row.subject_key or "") not in requested:
                    continue
                filters = {
                    "availability_as_of": availability_as_of,
                    "pit_mode": pit_mode or "SYSTEM",
                    "business_as_of": business_as_of,
                    "knowledge_as_of": knowledge_as_of,
                    "historical_projection": historical_projection,
                }
                if not self._pit_matches(payload, filters):
                    continue
                if not historical_projection and not self._lifecycle_matches(session, payload, filters):
                    continue
                try:
                    items.append(upgrade_signal_v5(payload))
                except ValueError:
                    # A legacy knowledge row has no complete v5 lineage and
                    # must not be emitted as a misleading signal.
                    continue
            return items


def _ticker_code(value: str) -> str | None:
    match = re.search(r"(?<!\d)(\d{6})(?!\d)", str(value))
    return match.group(1) if match else None


def _iso(value: object) -> str:
    if value is None:
        return ""
    text = value.isoformat() if isinstance(value, datetime) else str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def _iso_date(value: object) -> str:
    text = _iso(value)
    return text[:10]


def _binding_overlaps(binding: dict, target_start: object, target_end: object) -> bool:
    """Compare DATE/TIMESTAMP binding intervals, failing closed if unknown."""
    value_type = str(binding.get("value_type") or "").upper()
    if value_type not in {"DATE", "TIMESTAMP"}:
        # Some persisted projections omit value_type.  Infer it only from a
        # concrete endpoint; an unresolved binding must never match a range.
        if binding.get("start_time") or binding.get("end_time"):
            value_type = "TIMESTAMP"
        elif binding.get("start_date") or binding.get("end_date"):
            value_type = "DATE"
        else:
            return False
    if value_type == "DATE":
        def parse(value):
            return _date_value(value)

        lower = binding.get("start_date") or binding.get("earliest_start_date")
        upper = binding.get("end_date") or binding.get("latest_end_date")
    else:
        def parse(value):
            return _timestamp_value(value)

        lower = binding.get("start_time") or binding.get("earliest_start_time")
        upper = binding.get("end_time") or binding.get("latest_end_time")
    try:
        lower = parse(lower) if lower is not None else None
        upper = parse(upper) if upper is not None else None
        requested_lower = parse(target_start) if target_start is not None else None
        requested_upper = parse(target_end) if target_end is not None else None
    except (TypeError, ValueError):
        return False
    # No endpoints is UNRESOLVED/TIMELESS and cannot satisfy a target range.
    if lower is None and upper is None:
        return False
    return (requested_lower is None or upper is None or upper >= requested_lower) and (
        requested_upper is None or lower is None or lower <= requested_upper
    )


def _date_value(value: object):
    if value is None:
        return None
    text = value.isoformat() if hasattr(value, "isoformat") else str(value)
    return datetime.fromisoformat(text[:10]).date()


def _timestamp_value(value: object):
    if value is None:
        return None
    text = value.isoformat() if hasattr(value, "isoformat") else str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
