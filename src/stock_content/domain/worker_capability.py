"""Immutable worker capabilities and deterministic task routing."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class WorkerProfile(StrEnum):
    API = "api"
    CORE = "core"
    MEDIA = "media"
    MULTIMODAL = "multimodal"


class TaskKind(StrEnum):
    API = "api"
    CORE = "core"
    MEDIA = "media"
    OCR = "ocr"
    VISION = "vision"
    DIARIZATION = "diarization"
    MULTIMODAL = "multimodal"
    INDEX = "index"


@dataclass(frozen=True, slots=True)
class WorkerCapability:
    profile: WorkerProfile
    task_kinds: frozenset[TaskKind]
    ffmpeg: bool = False
    torch: bool = False
    cuda: bool = False

    def supports(self, task: TaskKind | str) -> bool:
        return TaskKind(task) in self.task_kinds


_CAPABILITIES: dict[WorkerProfile, WorkerCapability] = {
    WorkerProfile.API: WorkerCapability(WorkerProfile.API, frozenset({TaskKind.API})),
    WorkerProfile.CORE: WorkerCapability(WorkerProfile.CORE, frozenset({TaskKind.CORE, TaskKind.INDEX})),
    WorkerProfile.MEDIA: WorkerCapability(WorkerProfile.MEDIA, frozenset({TaskKind.MEDIA}), ffmpeg=True),
    WorkerProfile.MULTIMODAL: WorkerCapability(
        WorkerProfile.MULTIMODAL,
        frozenset({TaskKind.OCR, TaskKind.VISION, TaskKind.DIARIZATION, TaskKind.MULTIMODAL}),
        ffmpeg=True,
        torch=True,
        cuda=True,
    ),
}


def capability_for(profile: WorkerProfile | str) -> WorkerCapability:
    return _CAPABILITIES[WorkerProfile(profile)]


def route_task(task: TaskKind | str) -> WorkerProfile:
    """Return the only profile allowed to execute a task."""
    kind = TaskKind(task)
    for profile, capability in _CAPABILITIES.items():
        if capability.supports(kind):
            return profile
    raise ValueError(f"no worker capability for task {kind.value}")


def require_capability(profile: WorkerProfile | str, task: TaskKind | str) -> None:
    """Fail closed when a queue is accidentally assigned to another profile."""
    worker = capability_for(profile)
    if not worker.supports(task):
        raise ValueError(f"worker profile {worker.profile.value} cannot execute task {TaskKind(task).value}")


def capability_matrix() -> tuple[WorkerCapability, ...]:
    return tuple(_CAPABILITIES[profile] for profile in WorkerProfile)


__all__ = [
    "TaskKind", "WorkerCapability", "WorkerProfile", "capability_for", "capability_matrix",
    "require_capability", "route_task",
]
