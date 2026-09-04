"""Compatibility projection of source/media stages.

The concrete implementation remains in ``application.stages`` until the
characterization suite permits physical extraction without hash drift.
"""
from stock_content.application.stages import AudioStage, DownloadStage, FrameExtractionStage, ResolveSourceStage

__all__ = ["AudioStage", "DownloadStage", "FrameExtractionStage", "ResolveSourceStage"]
