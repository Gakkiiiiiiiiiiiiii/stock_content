"""Evidence-scoped equity mentions for local knowledge previews.

Displayed securities are observations, not claim subjects or recommendations.
This module does not infer a security from a company name or a nearby topic.
"""

from __future__ import annotations

from copy import deepcopy

SPOKEN_TIERS = {"FOCUSED_CHART_SPOKEN", "SLIDE_ENTITY_SPOKEN"}
VISUAL_ONLY_TIERS = {"FOCUSED_CHART_VISUAL_ONLY", "SLIDE_ENTITY_VISUAL_ONLY"}


def _canonical_label(name: str, code: str | None, market: str | None) -> str:
    qualifier = code or ("港股" if market == "HK" else market)
    return f"{name}（{qualifier}）" if qualifier else name


def _frame_id(frame: dict) -> str:
    value = frame.get("frame_id")
    if isinstance(value, str) and value.strip():
        return value
    return f"frame_{frame['image_sha256'][:24]}"


def _reproject_resolved_entity(
    card: dict,
    speech: list[dict],
    name: str,
    code: str | None,
    market: str | None,
    raw_entity_text: str | None,
    protected_indices: set[int],
    resolution_status: str,
) -> None:
    """Resolve only entity ambiguities whose audio coordinates were visually corroborated."""
    speech_indices = {item["segment_index"] for item in speech}
    for item in card.get("unresolved_items") or []:
        if (
            item.get("kind") == "ENTITY"
            and speech_indices.intersection(item.get("segment_indices") or [])
            and not protected_indices.intersection(item.get("segment_indices") or [])
            and (not raw_entity_text or item.get("raw_text") == raw_entity_text)
        ):
            canonical = _canonical_label(name, code, market)
            if item.get("status") == "UNRESOLVED":
                item["status"] = resolution_status
                item["resolution"] = canonical
    unresolved_entities = [
        item for item in card.get("unresolved_items") or []
        if item.get("kind") == "ENTITY" and item.get("status") == "UNRESOLVED"
    ]
    card["unresolved_entity_in_range"] = bool(unresolved_entities)
    if not unresolved_entities and "ENTITY_NAME_UNRESOLVED" in card.get("reason_codes", []):
        card["reason_codes"].remove("ENTITY_NAME_UNRESOLVED")


def _scope_visual_identity_without_resolving_speech(
    card: dict, eligible: list[dict], name: str, code: str | None, market: str | None = None
) -> None:
    ranges = [(stage["start_segment_index"], stage["end_segment_index"]) for stage in eligible]
    for item in card.get("unresolved_items") or []:
        if (
            item.get("kind") == "ENTITY"
            and item.get("status") == "UNRESOLVED"
            and any(
                start <= index <= end
                for index in item.get("segment_indices") or []
                for start, end in ranges
            )
        ):
            item["reason"] = (
                f"画面已确认规范身份为{_canonical_label(name, code, market)}；"
                "该原始口播与画面标的的关联仍未确认。"
            )
            item["resolution"] = f"画面规范身份：{_canonical_label(name, code, market)}；口播关联未决"


def link_equity_mentions(
    cards: list[dict], stages: list[dict], transcript_rows: list[dict], mentions: list[dict]
) -> tuple[list[dict], dict]:
    """Link reviewed mentions only when their declared stage contains source evidence.

    ``stages`` use the caller's IDs and zero-based topic indices. A mention can
    attach to one knowledge card at most; ambiguous or excluded scopes remain
    visible in the audit instead of being guessed into a card.
    """
    stage_by_id = {stage["stage_id"]: stage for stage in stages}
    if len(stage_by_id) != len(stages):
        raise ValueError("Duplicate stage ID")
    card_by_topic: dict[int, int] = {}
    linked_cards = deepcopy(cards)
    for card_index, card in enumerate(linked_cards):
        card["equity_mentions"] = []
        for topic_index in card["topic_indices"]:
            if topic_index in card_by_topic:
                raise ValueError("Topic belongs to multiple knowledge cards")
            card_by_topic[topic_index] = card_index

    unlinked: list[dict] = []
    linked_count = 0
    seen_ids: set[str] = set()
    visual_only_evidence_indices = {
        evidence["segment_index"]
        for mention in mentions
        if mention.get("evidence_tier") in VISUAL_ONLY_TIERS
        for evidence in mention.get("transcript_evidence") or []
        if isinstance(evidence.get("segment_index"), int)
    }
    for mention in mentions:
        entity_id = mention.get("entity_id")
        tier = mention.get("evidence_tier")
        identity = mention.get("identity_status")
        if not isinstance(entity_id, str) or not entity_id or entity_id in seen_ids:
            raise ValueError("Missing or duplicate equity entity ID")
        seen_ids.add(entity_id)
        if (tier, identity) not in {
            ("FOCUSED_CHART_SPOKEN", "CONFIRMED_IN_VIDEO"),
            ("FOCUSED_CHART_VISUAL_ONLY", "CONFIRMED_ON_SCREEN_SPEECH_LINK_UNRESOLVED"),
            ("SLIDE_ENTITY_SPOKEN", "CONFIRMED_IN_VIDEO"),
            ("SLIDE_ENTITY_VISUAL_ONLY", "CONFIRMED_ON_SCREEN_SPEECH_LINK_UNRESOLVED"),
        }:
            raise ValueError("Unsupported equity evidence status")
        code = mention.get("code")
        market = mention.get("market")
        if not isinstance(mention.get("name"), str) or not mention["name"].strip():
            raise ValueError("Equity name is missing")
        if tier.startswith("FOCUSED_CHART"):
            if not isinstance(code, str) or len(code) != 6 or not code.isascii() or not code.isdigit():
                raise ValueError("Focused-chart equity code is not six ASCII digits")
        elif code is not None and (
            not isinstance(code, str) or len(code) not in {5, 6} or not code.isascii() or not code.isdigit()
        ):
            raise ValueError("Slide entity code is invalid")
        if market is not None and (not isinstance(market, str) or not market.strip()):
            raise ValueError("Equity market is invalid")
        frames = mention.get("visual_evidence")
        if not isinstance(frames, list) or not frames:
            raise ValueError("Equity has no reviewed frame")
        for frame in frames:
            if not isinstance(frame.get("timestamp_ms"), int) or not isinstance(frame.get("image_sha256"), str):
                raise ValueError("Equity frame evidence is incomplete")
        speech = mention.get("transcript_evidence") or []
        if not isinstance(speech, list):
            raise ValueError("Equity transcript evidence is invalid")
        for evidence in speech:
            index = evidence.get("segment_index")
            if not isinstance(index, int) or not 0 <= index < len(transcript_rows):
                raise ValueError("Equity transcript coordinate is invalid")
            if evidence.get("text") != transcript_rows[index].get("text"):
                raise ValueError("Equity transcript quote differs from source")
        if tier in SPOKEN_TIERS and not speech:
            raise ValueError("Spoken equity has no transcript evidence")

        declared = mention.get("stage_ids")
        if not isinstance(declared, list) or not declared or any(stage_id not in stage_by_id for stage_id in declared):
            raise ValueError("Equity stage ID is invalid")
        eligible = []
        for stage_id in declared:
            stage = stage_by_id[stage_id]
            spoken_in_stage = any(
                stage["start_segment_index"] <= item["segment_index"] <= stage["end_segment_index"]
                for item in speech
            )
            speech_conflicts_with_stage = bool(speech) and not spoken_in_stage
            frame_in_stage = any(
                stage["start_ms"] <= frame["timestamp_ms"] <= stage["end_ms"]
                or (
                    spoken_in_stage
                    and stage["end_ms"] < frame["timestamp_ms"] <= stage["end_ms"] + 3000
                )
                for frame in frames
            )
            if (
                tier in SPOKEN_TIERS and spoken_in_stage
                or (
                    tier in VISUAL_ONLY_TIERS
                    and frame_in_stage
                    and not speech_conflicts_with_stage
                )
            ):
                eligible.append(stage)
        target_indices = {
            card_by_topic[stage["topic_index"]]
            for stage in eligible
            if stage["topic_index"] in card_by_topic
        }
        if len(target_indices) != 1:
            unlinked.append({
                "entity_id": entity_id,
                "reason": "AMBIGUOUS_KNOWLEDGE_SCOPE" if len(target_indices) > 1 else "NO_EVIDENCE_SCOPED_KNOWLEDGE",
            })
            continue
        card = linked_cards[target_indices.pop()]
        raw_spoken_mentions = [
            {"segment_index": item["segment_index"], "text": item["text"]} for item in speech
        ]
        speech_link_status = (
            "SPOKEN_AND_DISPLAYED_CONFIRMED"
            if tier in SPOKEN_TIERS else "DISPLAYED_ONLY_SPEECH_LINK_UNRESOLVED"
        )
        card["equity_mentions"].append({
            "entity_id": entity_id,
            "name": mention["name"],
            "code": code,
            "market": market,
            "code_status": mention.get("code_status") or (
                "CONFIRMED_IN_VIDEO" if code else "NOT_VISIBLE_IN_VIDEO"
            ),
            "evidence_tier": tier,
            "identity_status": identity,
            "link_status": speech_link_status,
            "raw_spoken_mentions": raw_spoken_mentions,
            "canonical_identity": {
                "name": mention["name"],
                "code": code,
                "market": market,
                "code_status": mention.get("code_status") or (
                    "CONFIRMED_IN_VIDEO" if code else "NOT_VISIBLE_IN_VIDEO"
                ),
                "identity_status": identity,
            },
            "speech_link_status": speech_link_status,
            "knowledge_relation": "TOPIC_SCOPED_MENTION_NOT_RECOMMENDATION",
            "stage_ids": [stage["stage_id"] for stage in eligible],
            "transcript_evidence": raw_spoken_mentions if tier in SPOKEN_TIERS else [],
            "visual_evidence": [
                {
                    "frame_id": _frame_id(frame),
                    "timestamp_ms": frame["timestamp_ms"],
                    "image_sha256": frame["image_sha256"],
                    "frame_scope_status": next(
                        (
                            "IN_DECLARED_STAGE"
                            if stage_by_id[stage_id]["start_ms"] <= frame["timestamp_ms"]
                            <= stage_by_id[stage_id]["end_ms"]
                            else "PRE_TOPIC_CORROBORATION"
                            if stage_by_id[stage_id]["start_ms"] - 3000 <= frame["timestamp_ms"]
                            < stage_by_id[stage_id]["start_ms"]
                            else "POST_TOPIC_CORROBORATION"
                        )
                        for stage_id in declared
                        if stage_by_id[stage_id]["start_ms"] - 3000 <= frame["timestamp_ms"]
                        <= stage_by_id[stage_id]["end_ms"] + 3000
                    ),
                }
                for frame in frames
            ],
            "recommendation_status": "NOT_A_RECOMMENDATION",
        })
        if tier in SPOKEN_TIERS:
            _reproject_resolved_entity(
                card,
                speech,
                mention["name"],
                code,
                market,
                mention.get("raw_entity_text"),
                visual_only_evidence_indices,
                (
                    "RESOLVED_BY_CROSS_MODAL"
                    if mention.get("asr_correction_supported") is True
                    else "RESOLVED_BY_VISUAL"
                ),
            )
        linked_count += 1
    return linked_cards, {
        "reviewed_mention_count": len(mentions),
        "linked_mention_count": linked_count,
        "cards_with_mentions": sum(bool(card["equity_mentions"]) for card in linked_cards),
        "unlinked": unlinked,
    }
