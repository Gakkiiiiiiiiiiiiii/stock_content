"""Reconcile reviewed ambiguities without regenerating knowledge viewpoints.

This deterministic local-preview stage applies audited ambiguity routing and
bounded audio corrections, re-runs structured time normalization, and finally
reprojects freshly reviewed visual equity identities. Existing viewpoints are
kept immutable through normalization; after linking, only stale entity-identity
phrases may be reconciled by the exact-coordinate evidence policy.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

from build_local_coherent_knowledge_preview import (
    corrected_transcript,
    json_bytes,
    project_cards,
    read_json,
    transcript_packet,
    validate_extraction,
    validate_topic_map,
    write_new,
)
from build_local_coherent_knowledge_preview_v2 import (
    normalize_card,
    project_literal_spoken_entities,
    reproject_candidate_entities,
    reviewed_equities,
    reviewed_equity_context,
    topic_stages,
)
from repair_local_coherent_knowledge_preview import editable_card

from stock_content.domain.ambiguity_resolution import (
    RESOLVED_REVIEW_STATUSES,
    ambiguity_item_id,
    card_has_high_risk_ambiguity,
    granular_unresolved_items,
)
from stock_content.domain.knowledge_equity_links import link_equity_mentions


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _deduplicated_items(items: list[dict]) -> list[dict]:
    pending = granular_unresolved_items(items)
    resolved: list[dict] = []
    seen: set[tuple] = set()
    for item in items:
        if item.get("status") not in RESOLVED_REVIEW_STATUSES:
            continue
        key = (
            item.get("kind"),
            tuple(sorted(set(item.get("segment_indices") or []))),
            str(item.get("raw_text") or ""),
            item.get("status"),
            str(item.get("resolution") or ""),
        )
        if key not in seen:
            resolved.append(copy.deepcopy(item))
            seen.add(key)
    return sorted(
        [*pending, *resolved],
        key=lambda item: (
            min(item.get("segment_indices") or [10**9]),
            item.get("status") != "UNRESOLVED",
            str(item.get("kind") or ""),
        ),
    )


def _apply_reviews(
    *, card: dict, knowledge_id: str, triage_by_id: dict[str, dict], audio_by_id: dict[str, dict]
) -> tuple[dict, dict[tuple, dict]]:
    result = copy.deepcopy(card)
    kept: list[dict] = []
    metadata: dict[tuple, dict] = {}
    for source_item in result.get("unresolved_items") or []:
        item = copy.deepcopy(source_item)
        if item.get("status") != "UNRESOLVED":
            kept.append(item)
            continue
        item_id = ambiguity_item_id(knowledge_id, item)
        decision = triage_by_id.get(item_id)
        if decision is None:
            kept.append(item)
            continue
        action = decision["action"]
        if action == "REMOVE_FALSE_POSITIVE":
            continue
        if decision.get("corrected_kind"):
            item["kind"] = decision["corrected_kind"]
        if decision.get("review_text"):
            item["raw_text"] = decision["review_text"]
        item["reason"] = decision["reason"]
        if action == "RESOLVED_BY_VIDEO_CONTEXT":
            item["status"] = "RESOLVED_BY_VIDEO_CONTEXT"
            item["resolution"] = decision["resolution"]
        audio = audio_by_id.get(item_id)
        if action == "AUDIO_REVIEW_REQUIRED" and audio:
            audio_status = audio.get("decision")
            if audio_status in {"RESOLVED_BY_AUDIO", "REUSED_EXISTING_REVIEWED_AUDIO_CORRECTION"}:
                item["status"] = "RESOLVED_BY_AUDIO"
                item["resolution"] = (
                    audio.get("candidate_text")
                    or decision.get("candidate_text")
                    or "双 ASR 与大模型复核已修正对应转录段"
                )
                item["reason"] = str(audio.get("reason") or "双 ASR 与大模型复核结果一致。")
            elif audio_status == "AUDIO_CANDIDATE_VISUAL_REQUIRED":
                item["reason"] = str(audio.get("reason") or decision["reason"])
            elif audio_status == "UNRESOLVED":
                item["reason"] = str(audio.get("reason") or decision["reason"])
        kept.append(item)
        metadata[
            (
                item.get("kind"),
                tuple(sorted(set(item.get("segment_indices") or []))),
                str(item.get("raw_text") or ""),
            )
        ] = {
            "ambiguity_item_id": item_id,
            "triage_action": action,
            "entity_type": decision.get("entity_type"),
            "candidate_text": (
                audio.get("candidate_text") if audio else decision.get("candidate_text")
            ),
        }
    result["unresolved_items"] = _deduplicated_items(kept)
    return result, metadata


def _attach_metadata(card: dict, metadata: dict[tuple, dict]) -> None:
    for item in card.get("unresolved_items") or []:
        key = (
            item.get("kind"),
            tuple(sorted(set(item.get("segment_indices") or []))),
            str(item.get("raw_text") or ""),
        )
        if key in metadata:
            item.update({key: value for key, value in metadata[key].items() if value is not None})


def _update_card_status(card: dict) -> None:
    active = [item for item in card.get("unresolved_items") or [] if item.get("status") == "UNRESOLVED"]
    high_risk = card_has_high_risk_ambiguity(card)
    mentions = card.get("equity_mentions") or []
    card["unresolved_entity_in_range"] = any(item.get("kind") == "ENTITY" for item in active)
    reason_codes = [
        code for code in card.get("reason_codes") or []
        if code not in {
            "VISUAL_RECHECK_PENDING", "ENTITY_NAME_UNRESOLVED", "TARGETED_ENTITY_REVIEW",
            "TARGETED_AMBIGUITY_REVIEW_COMPLETE", "TARGETED_AMBIGUITY_REVIEW_PARTIAL",
        }
    ]
    if mentions:
        card["visual_review_status"] = "TARGETED_ENTITY_FRAME_REVIEW_ONLY"
        reason_codes.append("TARGETED_ENTITY_REVIEW")
    else:
        card["visual_review_status"] = "TARGETED_AMBIGUITY_REVIEW_NO_EQUITY_FRAME"
    if high_risk:
        reason_codes.append("TARGETED_AMBIGUITY_REVIEW_PARTIAL")
        remaining = "、".join(sorted({str(item.get('kind')) for item in active}))
        card["status_reason"] = (
            f"新版未决项分诊及有边界复核已执行；仍保留无法由当前音频/画面可靠消除的"
            f"{remaining or '歧义'}。外部事实未核验。"
        )
    else:
        reason_codes.append("TARGETED_AMBIGUITY_REVIEW_COMPLETE")
        card["status_reason"] = (
            "新版未决项去重、分类、时间结构化与实体重投影已执行；"
            "画面代码仅在同期证据明确出现时保存，外部事实未核验。"
        )
    card["reason_codes"] = list(dict.fromkeys(reason_codes))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--triage-source-knowledge", type=Path, required=True)
    parser.add_argument("--triage", type=Path, required=True)
    parser.add_argument("--topic-map", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--audio-review", type=Path, required=True)
    parser.add_argument("--equity-review", type=Path, required=True)
    parser.add_argument("--corrected-transcript-output", type=Path, required=True)
    parser.add_argument("--opencc-package-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.opencc_package_dir))
    from opencc import OpenCC  # noqa: PLC0415

    draft, draft_hash = read_json(args.draft)
    triage_source, triage_source_hash = read_json(args.triage_source_knowledge)
    triage, triage_hash = read_json(args.triage)
    topic_map, map_hash = read_json(args.topic_map)
    source, source_hash = read_json(args.transcript)
    audio_review, audio_hash = read_json(args.audio_review)
    media_hash = topic_map["media_sha256"]
    if (
        triage.get("schema_version") != "local-ambiguity-triage.v1"
        or triage.get("audit", {}).get("passed") is not True
        or triage.get("source_knowledge_sha256") != triage_source_hash
        or audio_review.get("ambiguity_triage_sha256") != triage_hash
    ):
        raise ValueError("Ambiguity triage or audio-review provenance mismatch")
    transcript = corrected_transcript(source, source_hash, media_hash, audio_review)
    validate_topic_map(topic_map, transcript, source_hash, media_hash)
    packet = transcript_packet(topic_map, transcript)
    mentions, equity_hash = reviewed_equities(
        args.equity_review,
        map_hash=map_hash,
        source_hash=source_hash,
        media_hash=media_hash,
        audio_review_hash=audio_hash,
    )
    packet["reviewed_equities"] = reviewed_equity_context(
        mentions, topic_stages(topic_map, transcript)
    )

    triage_by_id = {item["item_id"]: item for item in triage.get("decisions") or []}
    audio_by_id = {
        item["item_id"]: item for item in audio_review.get("ambiguity_item_decisions") or []
    }
    source_ids = [card["knowledge_id"] for card in triage_source.get("knowledge") or []]
    if len(source_ids) != len(draft.get("knowledge") or []):
        raise ValueError("Reconciliation draft changed knowledge-card count")
    converter = OpenCC("t2s")
    candidate = {
        "knowledge": [],
        "excluded_topic_indices": copy.deepcopy(draft.get("excluded_topic_indices") or []),
    }
    metadata_by_position: list[dict[tuple, dict]] = []
    immutable_before: list[tuple] = []
    for position, projected_card in enumerate(draft["knowledge"]):
        editable = normalize_card(editable_card(projected_card, transcript), converter)
        immutable_before.append(tuple(editable.get(field) for field in (
            "knowledge_title", "atomic_statement", "detailed_explanation"
        )))
        reviewed, metadata = _apply_reviews(
            card=editable,
            knowledge_id=source_ids[position],
            triage_by_id=triage_by_id,
            audio_by_id=audio_by_id,
        )
        # Evidence coordinates are immutable; the literal quote follows the
        # newly corrected transcript row so the raw evidence contract remains valid.
        for evidence in reviewed.get("evidence") or []:
            evidence["quote"] = transcript["segments"][evidence["segment_index"]]["text"]
        candidate["knowledge"].append(reviewed)
        metadata_by_position.append(metadata)

    knowledge = validate_extraction(candidate, packet, transcript, structured=True)
    immutable_fields = ("knowledge_title", "atomic_statement", "detailed_explanation")
    for before, card in zip(immutable_before, knowledge, strict=True):
        # Time normalization may modernize legacy explanatory wording. Restore
        # reviewed prose before the later exact-coordinate identity-only pass.
        for field, value in zip(immutable_fields, before, strict=True):
            card[field] = value
    reproject_candidate_entities(knowledge, packet)
    projected = project_cards(
        knowledge, topic_map, transcript, converter, map_hash, source_hash, media_hash
    )
    project_literal_spoken_entities(projected, packet)
    for position, card in enumerate(projected):
        card["knowledge_id"] = source_ids[position]
        card["unresolved_items"] = _deduplicated_items(card.get("unresolved_items") or [])
        _attach_metadata(card, metadata_by_position[position])
    projected, equity_link_audit = link_equity_mentions(
        projected,
        topic_stages(topic_map, transcript),
        transcript["segments"],
        mentions,
    )
    for card in projected:
        _update_card_status(card)

    if not args.corrected_transcript_output.exists():
        corrected_hash = write_new(args.corrected_transcript_output, transcript)
    else:
        _, corrected_hash = read_json(args.corrected_transcript_output)
        if args.corrected_transcript_output.read_bytes() != json_bytes(transcript):
            raise ValueError("Existing corrected transcript differs")
    output = {
        "schema_version": "local-coherent-knowledge.v1",
        "status": "LOCAL_REVIEW_ONLY_NOT_SEALED_OR_PERSISTED",
        "run_kind": "LOCAL_REPARSE_PREVIEW_ENTITY_PROSE_RECONCILIATION",
        "workflow_run_kind": "LOCAL_REPARSE_PREVIEW",
        "requested_model": "gpt-6-sol",
        "model_identity_source": "accepted_codex_cli_invocation_not_response_metadata",
        "prompt_version": (
            "ambiguity-reconciliation.v2.deterministic-with-entity-prose-consistency"
        ),
        "topic_map_sha256": map_hash,
        "source_transcript_sha256": source_hash,
        "corrected_transcript_sha256": corrected_hash,
        "audio_review_sha256": audio_hash,
        "ambiguity_triage_sha256": triage_hash,
        "triage_source_knowledge_sha256": triage_source_hash,
        "source_draft_sha256": draft_hash,
        "media_sha256": media_hash,
        "external_fact_verification": "NOT_PERFORMED",
        "layer_freshness": {
            "knowledge_viewpoints": "REUSED_HASH_VERIFIED",
            "ambiguity_triage": "REUSED_AUDITED_GPT6_SOL",
            "audio_review": "REUSED_HASH_VERIFIED",
            "equity_visual_review": "REUSED_AUDITED_GPT6_SOL",
            "entity_prose_reconciliation": "FRESH_DETERMINISTIC",
        },
        "visual_review_status": "TARGETED_AMBIGUITY_AND_ENTITY_REVIEW",
        "equity_review_sha256": equity_hash,
        "equity_link_audit": equity_link_audit,
        "source_topic_count": len(topic_map["segments"]),
        "excluded_topic_indices": candidate["excluded_topic_indices"],
        "excluded_topic_notes": copy.deepcopy(draft.get("excluded_topic_notes") or []),
        "knowledge_count": len(projected),
        "citation_repairs": candidate.get("citation_repairs", []),
        "knowledge": projected,
        "audit": {
            "passed": True,
            "method": (
                "deterministic schema/provenance validation after independently audited GPT-6 Sol "
                "ambiguity triage, dual-ASR adjudication, and fresh visual review"
            ),
            "issues": [],
            "entity_prose_consistency": {
                "passed": True,
                "policy": "EXACT_SPOKEN_COORDINATE_AND_CONFIRMED_DISPLAY_ONLY",
                "reconciled_card_count": sum(
                    bool(card.get("entity_prose_reconciliation")) for card in projected
                ),
            },
        },
    }
    output_hash = write_new(args.output, output)
    print(json.dumps({
        "status": "PASS_LOCAL_REVIEW_ONLY",
        "knowledge_count": len(projected),
        "active_high_risk_cards": sum(card_has_high_risk_ambiguity(card) for card in projected),
        "output_sha256": output_hash,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
