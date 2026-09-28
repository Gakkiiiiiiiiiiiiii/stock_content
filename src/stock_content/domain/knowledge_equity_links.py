"""Evidence-scoped equity mentions for local knowledge previews.

Displayed securities are observations, not claim subjects or recommendations.
This module does not infer a security from a company name or a nearby topic.
"""

from __future__ import annotations

from copy import deepcopy


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
        }:
            raise ValueError("Unsupported equity evidence status")
        code = mention.get("code")
        if not isinstance(mention.get("name"), str) or not mention["name"].strip():
            raise ValueError("Equity name is missing")
        if not isinstance(code, str) or len(code) != 6 or not code.isascii() or not code.isdigit():
            raise ValueError("Equity code is not six ASCII digits")
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
        if tier == "FOCUSED_CHART_SPOKEN" and not speech:
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
            frame_in_stage = any(
                stage["start_ms"] <= frame["timestamp_ms"] <= stage["end_ms"]
                for frame in frames
            )
            if spoken_in_stage or (not speech and tier == "FOCUSED_CHART_VISUAL_ONLY" and frame_in_stage):
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
        card["equity_mentions"].append({
            "entity_id": entity_id,
            "name": mention["name"],
            "code": code,
            "evidence_tier": tier,
            "identity_status": identity,
            "link_status": (
                "SPOKEN_AND_DISPLAYED_CONFIRMED"
                if tier == "FOCUSED_CHART_SPOKEN" else "DISPLAYED_ONLY_SPEECH_LINK_UNRESOLVED"
            ),
            "knowledge_relation": "TOPIC_SCOPED_MENTION_NOT_RECOMMENDATION",
            "stage_ids": [stage["stage_id"] for stage in eligible],
            "transcript_evidence": [
                {"segment_index": item["segment_index"], "text": item["text"]} for item in speech
            ],
            "visual_evidence": [
                {"timestamp_ms": frame["timestamp_ms"], "image_sha256": frame["image_sha256"]}
                for frame in frames
            ],
            "recommendation_status": "NOT_A_RECOMMENDATION",
        })
        linked_count += 1
    return linked_cards, {
        "reviewed_mention_count": len(mentions),
        "linked_mention_count": linked_count,
        "cards_with_mentions": sum(bool(card["equity_mentions"]) for card in linked_cards),
        "unlinked": unlinked,
    }
