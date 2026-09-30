"""Evidence-scoped equity mentions for local knowledge previews.

Displayed securities are observations, not claim subjects or recommendations.
This module does not infer a security from a company name or a nearby topic.
"""

from __future__ import annotations

import re
from copy import deepcopy

SPOKEN_TIERS = {"FOCUSED_CHART_SPOKEN", "SLIDE_ENTITY_SPOKEN"}
VISUAL_ONLY_TIERS = {"FOCUSED_CHART_VISUAL_ONLY", "SLIDE_ENTITY_VISUAL_ONLY"}
VISUAL_ONLY_CORROBORATION_GRACE_MS = 15_000
PROSE_FIELDS = (
    "knowledge_title",
    "atomic_statement",
    "detailed_explanation",
    "subject",
    "applicability",
    "risks",
    "invalidation_conditions",
)
RESOLVED_VISUAL_ENTITY_STATUSES = {"RESOLVED_BY_VISUAL", "RESOLVED_BY_CROSS_MODAL"}
IDENTITY_UNCERTAINTY_RE = re.compile(
    r"身份(?:未核实|未确认|尚未确认|无法确认|未经确认|未获核实)"
    r"|(?:无法|不能|不足以)确认(?:其对应股票|公司身份|具体公司|具体标的|股票代码)"
    r"|不能据此确认为(?:某家)?上市公司"
    r"|(?:转录|转写|口播)?名称(?:未确认|尚未确认)"
)
GENERIC_IDENTITY_PLACEHOLDER_RE = re.compile(
    r"(?:该)?身份(?:未核实|未确认|尚未确认)(?:的)?标的"
    r"|(?:转录|转写|口播)名称(?:未确认|尚未确认)的[^，。；！？]{0,16}标的"
)


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
    normalized_raw_entity = (raw_entity_text or "").strip()
    for item in card.get("unresolved_items") or []:
        item_raw_text = str(item.get("raw_text") or "").strip()
        raw_text_matches = (
            not normalized_raw_entity
            or item_raw_text == normalized_raw_entity
            or normalized_raw_entity in item_raw_text
            or item_raw_text in normalized_raw_entity
        )
        if (
            item.get("kind") == "ENTITY"
            and speech_indices.intersection(item.get("segment_indices") or [])
            and not protected_indices.intersection(item.get("segment_indices") or [])
            and raw_text_matches
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


def _short_raw_entity(value: object) -> str | None:
    raw = str(value or "").strip()
    if not raw or len(raw) > 32 or re.search(r"[，。；！？\n]", raw):
        return None
    return raw


def _confirmed_prose_targets(card: dict) -> list[dict]:
    """Return visually resolved entities whose exact spoken coordinates match the card review."""
    targets: list[dict] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for mention in card.get("equity_mentions") or []:
        if mention.get("speech_link_status") != "SPOKEN_AND_DISPLAYED_CONFIRMED":
            continue
        speech_indices = {
            item.get("segment_index")
            for item in mention.get("raw_spoken_mentions") or []
            if isinstance(item.get("segment_index"), int)
        }
        if not speech_indices:
            continue
        identity = mention.get("canonical_identity") or {}
        name = str(identity.get("name") or mention.get("name") or "").strip()
        if not name:
            continue
        label = _canonical_label(
            name,
            identity.get("code", mention.get("code")),
            identity.get("market", mention.get("market")),
        )
        for item in card.get("unresolved_items") or []:
            item_indices = {
                index for index in item.get("segment_indices") or [] if isinstance(index, int)
            }
            resolution = str(item.get("resolution") or "")
            if (
                item.get("kind") != "ENTITY"
                or item.get("status") not in RESOLVED_VISUAL_ENTITY_STATUSES
                or not speech_indices.intersection(item_indices)
                or name not in resolution
            ):
                continue
            key = (str(mention.get("entity_id") or name), tuple(sorted(speech_indices)))
            if key in seen:
                continue
            seen.add(key)
            targets.append({
                "entity_id": mention.get("entity_id"),
                "name": name,
                "label": label,
                "code": identity.get("code", mention.get("code")),
                "raw_text": _short_raw_entity(item.get("raw_text")),
                "segment_indices": sorted(speech_indices.intersection(item_indices)),
            })
    return targets


def _identity_confirmation_sentence(target: dict) -> str:
    raw = target.get("raw_text")
    raw_prefix = f"本卡原始口播为“{raw}”，" if raw else ""
    code_note = "" if target.get("code") else "；视频画面未显示股票代码"
    return f"{raw_prefix}同期画面确认其对应{target['label']}{code_note}"


def _rewrite_entity_identity_prose(
    text: str,
    target: dict,
    *,
    allow_generic: bool,
    allow_scoped_pronoun: bool,
) -> str:
    raw = target.get("raw_text")
    label = target["label"]
    result = text
    if raw:
        raw_sentinel = "\x00RAW_SPOKEN_ENTITY\x00"
        scoped_parenthetical = re.compile(
            rf"(?:转录|转写)作[“\"]{re.escape(raw)}[”\"]的标的"
            r"[（(](?:公司)?身份(?:未核实|未确认|尚未确认)[）)]"
        )
        result = scoped_parenthetical.sub(label, result)
        leading_descriptor = re.compile(
            rf"身份(?:未核实|未确认|尚未确认)[、，]"
            rf"名称(?:转录|转写)为[“\"]{re.escape(raw)}[”\"]的"
        )
        result = leading_descriptor.sub(f"{label}的", result)
        trailing_descriptor = re.compile(
            rf"(?:转录|转写)中的[“\"]{re.escape(raw)}[”\"]"
            r"身份(?:未核实|未确认|尚未确认)"
        )
        result = trailing_descriptor.sub(
            f"同期画面已确认其规范身份为{label}", result
        )
        identity_descriptor = re.compile(
            rf"(?:口播|转录|转写)名称为[“\"]{re.escape(raw)}[”\"]"
            r"[、，]身份(?:未核实|未确认|尚未确认)(?:的)?标的"
        )
        result = identity_descriptor.sub(label, result)
        spoken_placeholder = re.compile(
            rf"(?:名称)?转录(?:为|作)[“\"]{re.escape(raw)}[”\"]的标的"
        )
        result = spoken_placeholder.sub(label, result)
        result = result.replace(f"一个{label}", label)
        scoped_placeholder = re.compile(
            rf"(?:某)?(?:名称)?(?:转录|转写)为[“\"]{re.escape(raw)}[”\"]"
            r"[、，]?(?:具体)?身份(?:未核实|未确认|尚未确认|无法确认|未经确认)的?标的"
        )
        result = scoped_placeholder.sub(label, result)
        identity_only_sentence = re.compile(
            rf"(?:本卡|原始)?口播(?:名称)?(?:为|是)?[“\"]?{re.escape(raw)}[”\"]?"
            r"[^。！？]*(?:无法|不能|未能)[^。！？]*(?:身份|公司|股票|标的)[^。！？]*"
        )
        protected_target = {**target, "raw_text": raw_sentinel}
        result = identity_only_sentence.sub(
            _identity_confirmation_sentence(protected_target), result
        )
        result = result.replace(raw, target["name"])
        result = result.replace(raw_sentinel, raw)
    if allow_scoped_pronoun:
        confirmation = f"同期画面已确认其规范身份为{label}"
        result = re.sub(r"该名称(?:仍)?不足以确认公司身份", confirmation, result)
        result = re.sub(
            r"该标的身份及([^。；]+)未获核实",
            r"该标的身份已由同期画面确认，\1仍未获核实",
            result,
        )
        result = re.sub(
            rf"[“\"]?{re.escape(target['name'])}[”\"]?也?不能据此确认为(?:某家)?上市公司",
            confirmation,
            result,
        )
        result = result.replace(
            "两个身份未确认的标的",
            f"{label}与另一个身份未确认的标的",
        )
        result = result.replace(
            "标的及术语得到进一步核实",
            "相关术语得到进一步核实",
        )
        result = result.replace(
            "具体证券身份和压力位未确认",
            "具体证券身份已由同期画面确认，压力位仍未确认",
        )
    if allow_generic:
        result = GENERIC_IDENTITY_PLACEHOLDER_RE.sub(label, result)
        result = re.sub(
            r"其(?:公司|标的)?身份(?:尚)?(?:无法|未能|未|未经)(?:得到)?确认",
            f"其规范身份经同期画面确认为{label}",
            result,
        )
        result = re.sub(
            r"(?:仅凭本卡内容)?(?:无法|不能)确认公司身份或股票代码",
            f"同期画面已确认其规范身份为{label}"
            + ("" if target.get("code") else "，但视频画面未显示股票代码"),
            result,
        )
    return result


def _identity_uncertainty_near(text: str, value: str, *, distance: int = 24) -> bool:
    if not value:
        return False
    occurrences = [match.span() for match in re.finditer(re.escape(value), text)]
    for uncertainty in IDENTITY_UNCERTAINTY_RE.finditer(text):
        for start, end in occurrences:
            if end >= uncertainty.start() - distance and start <= uncertainty.end() + distance:
                return True
    return False


def validate_confirmed_entity_prose_consistency(cards: list[dict]) -> None:
    """Fail closed when a coordinate-matched confirmed entity still has stale identity prose."""
    issues: list[str] = []
    for card in cards:
        targets = _confirmed_prose_targets(card)
        unresolved_entities = any(
            item.get("kind") == "ENTITY" and item.get("status") == "UNRESOLVED"
            for item in card.get("unresolved_items") or []
        )
        allow_generic = len(targets) == 1 and not unresolved_entities
        for target in targets:
            raw = target.get("raw_text")
            for field in PROSE_FIELDS:
                text = card.get(field)
                if not isinstance(text, str) or not IDENTITY_UNCERTAINTY_RE.search(text):
                    continue
                field_allows_generic = allow_generic or (
                    field == "knowledge_title"
                    and len(targets) == 1
                    and not re.search(r"多个|两个|若干|部分标的", text)
                )
                comparison = text.replace(
                    f"{target['label']}与另一个身份未确认的标的",
                    f"{target['label']}与另一待核标的",
                )
                sentences = re.split(r"(?<=[。！？；])", comparison)
                stale = any(
                    IDENTITY_UNCERTAINTY_RE.search(sentence)
                    and (
                        _identity_uncertainty_near(sentence, target["name"])
                        or bool(raw and _identity_uncertainty_near(sentence, raw))
                        or bool(
                            field_allows_generic
                            and GENERIC_IDENTITY_PLACEHOLDER_RE.search(sentence)
                        )
                    )
                    for sentence in sentences
                )
                if stale:
                    issues.append(
                        f"{card.get('knowledge_id', '<unknown>')}:{field}:{target['name']}"
                    )
    if issues:
        raise ValueError(
            "Confirmed entity identity remains unresolved in prose: " + ", ".join(issues)
        )


def reconcile_confirmed_entity_prose(cards: list[dict]) -> list[dict]:
    """Narrowly rewrite stale identity placeholders after visual entity projection.

    This never changes transcript evidence or ``raw_spoken_mentions`` and only
    acts on entities whose resolved ambiguity item shares an exact spoken
    segment with a ``SPOKEN_AND_DISPLAYED_CONFIRMED`` mention.
    """
    for card in cards:
        targets = _confirmed_prose_targets(card)
        unresolved_entities = any(
            item.get("kind") == "ENTITY" and item.get("status") == "UNRESOLVED"
            for item in card.get("unresolved_items") or []
        )
        allow_generic = len(targets) == 1 and not unresolved_entities
        allow_scoped_pronoun = len(targets) == 1
        reconciliation: list[dict] = []
        for target in targets:
            changed_fields: list[str] = []
            for field in PROSE_FIELDS:
                before = card.get(field)
                if not isinstance(before, str):
                    continue
                field_allows_generic = allow_generic or (
                    field == "knowledge_title"
                    and len(targets) == 1
                    and not re.search(r"多个|两个|若干|部分标的", before)
                )
                after = _rewrite_entity_identity_prose(
                    before,
                    target,
                    allow_generic=field_allows_generic,
                    allow_scoped_pronoun=allow_scoped_pronoun,
                )
                if after != before:
                    card[field] = after
                    changed_fields.append(field)
            if changed_fields:
                reconciliation.append({
                    "entity_id": target.get("entity_id"),
                    "canonical_name": target["name"],
                    "raw_spoken_text": target.get("raw_text"),
                    "segment_indices": target["segment_indices"],
                    "changed_fields": changed_fields,
                    "policy": "EXACT_SPOKEN_COORDINATE_AND_CONFIRMED_DISPLAY_ONLY",
                })
        if reconciliation:
            card["entity_prose_reconciliation"] = reconciliation
    validate_confirmed_entity_prose_consistency(cards)
    return cards


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
                    not speech_conflicts_with_stage
                    and stage["end_ms"] < frame["timestamp_ms"]
                    <= stage["end_ms"] + VISUAL_ONLY_CORROBORATION_GRACE_MS
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
                        <= stage_by_id[stage_id]["end_ms"] + VISUAL_ONLY_CORROBORATION_GRACE_MS
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
    reconcile_confirmed_entity_prose(linked_cards)
    return linked_cards, {
        "reviewed_mention_count": len(mentions),
        "linked_mention_count": linked_count,
        "cards_with_mentions": sum(bool(card["equity_mentions"]) for card in linked_cards),
        "unlinked": unlinked,
    }
