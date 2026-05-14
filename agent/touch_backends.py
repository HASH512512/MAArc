from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from algo.algo_base import TouchAction


class TouchBackend(Protocol):
    name: str

    def dispatch(self, event) -> dict[str, int | str | None]: ...

    def release_all(self) -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class PointerMapper:
    max_contacts: int = 10
    _pointer_to_contact: dict[int, int] = field(default_factory=dict)

    def down(self, pointer: int) -> int:
        if pointer in self._pointer_to_contact:
            return self._pointer_to_contact[pointer]
        used = set(self._pointer_to_contact.values())
        for contact in range(self.max_contacts):
            if contact not in used:
                self._pointer_to_contact[pointer] = contact
                return contact
        raise RuntimeError(
            f"No Maa contact available for pointer {pointer}; active={self._pointer_to_contact}"
        )

    def get(self, pointer: int) -> int:
        if pointer not in self._pointer_to_contact:
            raise RuntimeError(f"Pointer {pointer} is not active")
        return self._pointer_to_contact[pointer]

    def up(self, pointer: int) -> int:
        contact = self.get(pointer)
        del self._pointer_to_contact[pointer]
        return contact

    def active_contacts(self) -> list[int]:
        return sorted(self._pointer_to_contact.values())


class MaaTouchBackend:
    name = "maa"

    def __init__(self, controller, wait_mode: str = "wait_each", max_contacts: int = 10) -> None:
        if wait_mode not in {"wait_each", "batch_tick", "none"}:
            raise ValueError(f"Unsupported Maa wait mode: {wait_mode}")
        self.controller = controller
        self.wait_mode = wait_mode
        self.mapper = PointerMapper(max_contacts=max_contacts)
        self._pending_jobs = []

    def dispatch(self, event) -> dict[str, int | str | None]:
        x, y = event.pos
        action = event.action
        pointer = int(event.pointer)
        contact: int | None
        if action in {TouchAction.DOWN, TouchAction.POINTER_DOWN}:
            contact = self.mapper.down(pointer)
            job = self.controller.post_touch_down(int(x), int(y), contact=contact, pressure=1)
        elif action is TouchAction.MOVE:
            contact = self.mapper.get(pointer)
            job = self.controller.post_touch_move(int(x), int(y), contact=contact, pressure=1)
        elif action in {TouchAction.UP, TouchAction.POINTER_UP, TouchAction.CANCEL}:
            contact = self.mapper.up(pointer)
            job = self.controller.post_touch_up(contact=contact)
        else:
            return {"backend": self.name, "contact": None, "action": action.name}

        if self.wait_mode == "wait_each":
            job.wait()
        elif self.wait_mode == "batch_tick":
            self._pending_jobs.append(job)
        return {"backend": self.name, "contact": contact, "action": action.name}

    def flush_tick(self) -> None:
        if not self._pending_jobs:
            return
        for job in self._pending_jobs:
            job.wait()
        self._pending_jobs.clear()

    def release_all(self) -> None:
        for contact in self.mapper.active_contacts():
            try:
                job = self.controller.post_touch_up(contact=contact)
                if self.wait_mode != "none":
                    job.wait()
            except Exception as exc:
                print(f"[WARN] Failed to release Maa contact {contact}: {exc}")

    def close(self) -> None:
        self.release_all()


class ScrcpyTouchBackend:
    name = "scrcpy"

    def __init__(
        self,
        adb_serial: str | None = None,
        max_fps: int = 60,
        video_bit_rate: int | None = None,
        video_crop: tuple[int, int, int, int] | None = None,
    ) -> None:
        from control import DeviceController

        self.adb_serial = adb_serial
        self.controller = DeviceController(
            serial=adb_serial,
            max_fps=max_fps,
            video_bit_rate=video_bit_rate,
            video_crop=video_crop,
        )
        self._active_pointers: dict[int, tuple[int, int]] = {}

    def dispatch(self, event) -> dict[str, int | str | None]:
        x, y = event.pos
        self.controller.touch(int(x), int(y), event.action, int(event.pointer))
        if event.action in {TouchAction.DOWN, TouchAction.MOVE, TouchAction.POINTER_DOWN}:
            self._active_pointers[int(event.pointer)] = (int(x), int(y))
        elif event.action in {TouchAction.UP, TouchAction.POINTER_UP, TouchAction.CANCEL}:
            self._active_pointers.pop(int(event.pointer), None)
        return {"backend": self.name, "contact": int(event.pointer), "action": event.action.name}

    def release_all(self) -> None:
        for pointer, (x, y) in list(self._active_pointers.items()):
            try:
                self.controller.touch(x, y, TouchAction.UP, pointer)
            except Exception as exc:
                print(f"[WARN] Failed to release scrcpy pointer {pointer}: {exc}")
        self._active_pointers.clear()

    def close(self) -> None:
        self.release_all()


def create_touch_backend(
    name: str,
    context,
    maa_wait_mode: str = "wait_each",
    adb_serial: str | None = None,
    scrcpy_max_fps: int = 60,
    scrcpy_video_bit_rate: int | None = None,
    scrcpy_video_crop: tuple[int, int, int, int] | None = None,
) -> TouchBackend:
    normalized = name.strip().lower()
    if normalized == "maa":
        return MaaTouchBackend(context.tasker.controller, wait_mode=maa_wait_mode)
    if normalized == "scrcpy":
        return ScrcpyTouchBackend(
            adb_serial=adb_serial,
            max_fps=scrcpy_max_fps,
            video_bit_rate=scrcpy_video_bit_rate,
            video_crop=scrcpy_video_crop,
        )
    raise ValueError(f"Unsupported input backend: {name}")
