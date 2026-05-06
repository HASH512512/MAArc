from __future__ import annotations


class TouchEvent:
    def __init__(
        self,
        position,
        action,
        pointer,
        alpha=1.0,
        source_note_id: int | None = None,
        source_type: str | None = None,
        logical_tick: int | None = None,
        logical_pos: tuple[float, float] | None = None,
    ):
        self.position = position
        self.pos = position
        self.action = action
        self.pointer = pointer
        self.alpha = alpha
        self.source_note_id = source_note_id
        self.source_type = source_type
        self.logical_tick = logical_tick
        self.logical_pos = logical_pos
