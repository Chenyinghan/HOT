"""Shared evaluator-supply primitives for dynamic and static search."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Deque,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Optional,
    Tuple,
    TypeVar,
)


T = TypeVar("T")


class ReadyReservoir(Generic[T]):
    """Bounded scheduler READY storage with common refill semantics."""

    def __init__(self, low_watermark: int = 0, high_watermark: int = 1) -> None:
        self.low_watermark = max(0, int(low_watermark))
        self.high_watermark = max(1, int(high_watermark))
        if self.low_watermark > self.high_watermark:
            raise ValueError("READY low watermark must not exceed high watermark")
        self._items: Deque[T] = deque()

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def append(self, item: T) -> None:
        self._items.append(item)

    def extend(self, items: Iterable[T]) -> None:
        self._items.extend(items)

    def popleft(self) -> T:
        return self._items.popleft()

    def clear(self) -> None:
        self._items.clear()

    @property
    def needs_refill(self) -> bool:
        return len(self) < self.low_watermark

    @property
    def refill_deficit(self) -> int:
        return max(0, self.high_watermark - len(self))


@dataclass
class SchedulerOccupancyMonitor:
    """Time-weighted evaluator occupancy with persistent warning hysteresis."""

    target_workers: int
    low_fraction: float
    recovery_fraction: float
    warning_duration_seconds: float
    recovery_duration_seconds: float
    underfill_started_at: Optional[float] = None
    recovered_started_at: Optional[float] = None
    warning_active: bool = False
    idle_worker_seconds: float = 0.0
    active_worker_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    previous_time: Optional[float] = None
    previous_active: int = 0
    occupancy_min: float = 1.0
    warning_count: int = 0
    recovery_count: int = 0
    max_underfill_duration_seconds: float = 0.0

    def observe(
        self,
        active: int,
        *,
        submission_open: bool,
        now: Optional[float] = None,
    ) -> Optional[str]:
        timestamp = time.monotonic() if now is None else float(now)
        active = min(self.target_workers, max(0, int(active)))
        if self.previous_time is not None:
            elapsed = max(0.0, timestamp - self.previous_time)
            self.elapsed_seconds += elapsed
            self.idle_worker_seconds += elapsed * max(
                0, self.target_workers - self.previous_active
            )
            self.active_worker_seconds += elapsed * self.previous_active
        self.previous_time = timestamp
        self.previous_active = active
        fraction = active / float(max(1, self.target_workers))
        self.occupancy_min = min(self.occupancy_min, fraction)
        if not submission_open:
            self.underfill_started_at = None
            self.recovered_started_at = None
            return None
        if fraction < self.low_fraction:
            self.recovered_started_at = None
            if self.underfill_started_at is None:
                self.underfill_started_at = timestamp
            duration = timestamp - self.underfill_started_at
            self.max_underfill_duration_seconds = max(
                self.max_underfill_duration_seconds, duration
            )
            if (
                not self.warning_active
                and duration >= self.warning_duration_seconds
            ):
                self.warning_active = True
                self.warning_count += 1
                return "warning"
            return None
        if fraction >= self.recovery_fraction:
            self.underfill_started_at = None
            if self.warning_active:
                if self.recovered_started_at is None:
                    self.recovered_started_at = timestamp
                if (
                    timestamp - self.recovered_started_at
                    >= self.recovery_duration_seconds
                ):
                    self.warning_active = False
                    self.recovered_started_at = None
                    self.recovery_count += 1
                    return "recovery"
            return None
        self.recovered_started_at = None
        return None

    def diagnostics(self) -> Dict[str, Any]:
        mean = (
            self.active_worker_seconds
            / (self.elapsed_seconds * float(max(1, self.target_workers)))
            if self.elapsed_seconds > 0.0
            else 0.0
        )
        current_underfill = (
            max(0.0, time.monotonic() - self.underfill_started_at)
            if self.underfill_started_at is not None
            else 0.0
        )
        return {
            "scheduler_occupancy_fraction": (
                self.previous_active / float(max(1, self.target_workers))
            ),
            "scheduler_occupancy_min": float(self.occupancy_min),
            "scheduler_occupancy_mean": float(mean),
            "scheduler_idle_worker_seconds": float(self.idle_worker_seconds),
            "scheduler_underfill_duration_seconds": float(current_underfill),
            "scheduler_underfill_duration_max_seconds": float(
                self.max_underfill_duration_seconds
            ),
            "scheduler_supply_warning_count": int(self.warning_count),
            "scheduler_supply_recovery_count": int(self.recovery_count),
            "scheduler_supply_warning_active": bool(self.warning_active),
        }


@dataclass
class SchedulerSupplyCriticalGuard:
    """Abort only sustained, productive, real-evaluator underfill."""

    target_workers: int
    critical_fraction: float
    critical_seconds: float
    recent_new_seconds: float
    rate_window_seconds: float
    started_at: float = field(default_factory=time.monotonic)
    underfill_started_at: Optional[float] = None
    last_new_at: Optional[float] = None
    new_events: Deque[Tuple[float, int]] = field(default_factory=deque)
    proposal_events: Deque[Tuple[float, int]] = field(default_factory=deque)
    repeat_events: Deque[Tuple[float, int]] = field(default_factory=deque)
    invalid_events: Deque[Tuple[float, int]] = field(default_factory=deque)
    cache_events: Deque[Tuple[float, int]] = field(default_factory=deque)
    completion_events: Deque[Tuple[float, int]] = field(default_factory=deque)
    triggered: bool = False

    def _prune(self, events: Deque[Tuple[float, int]], now: float) -> None:
        cutoff = float(now) - self.rate_window_seconds
        while events and events[0][0] < cutoff:
            events.popleft()

    def _record(
        self,
        events: Deque[Tuple[float, int]],
        count: int,
        now: float,
    ) -> None:
        if count > 0:
            events.append((float(now), int(count)))
        self._prune(events, now)

    def record_new(self, count: int, *, now: Optional[float] = None) -> None:
        timestamp = time.monotonic() if now is None else float(now)
        if count > 0:
            self.last_new_at = timestamp
        self._record(self.new_events, count, timestamp)

    def record_proposals(
        self,
        *,
        total: int,
        novel: int,
        repeat: int,
        invalid: int,
        cache: int,
        now: Optional[float] = None,
    ) -> None:
        timestamp = time.monotonic() if now is None else float(now)
        self._record(self.proposal_events, total, timestamp)
        self.record_new(novel, now=timestamp)
        self._record(self.repeat_events, repeat, timestamp)
        self._record(self.invalid_events, invalid, timestamp)
        self._record(self.cache_events, cache, timestamp)

    def record_completion(self, count: int, *, now: Optional[float] = None) -> None:
        timestamp = time.monotonic() if now is None else float(now)
        self._record(self.completion_events, count, timestamp)

    def defer(self, *, now: Optional[float] = None) -> None:
        timestamp = time.monotonic() if now is None else float(now)
        self.triggered = False
        self.underfill_started_at = timestamp

    def observe(
        self,
        *,
        active_futures: int,
        ready_depth: int,
        submission_open: bool,
        root_exhausted: bool,
        now: Optional[float] = None,
    ) -> bool:
        timestamp = time.monotonic() if now is None else float(now)
        recently_new = (
            self.last_new_at is not None
            and timestamp - self.last_new_at <= self.recent_new_seconds
        )
        critical = (
            self.critical_seconds > 0.0
            and submission_open
            and not root_exhausted
            and int(ready_depth) == 0
            and int(active_futures)
            < self.critical_fraction * float(self.target_workers)
            and recently_new
        )
        if not critical:
            self.underfill_started_at = None
            return False
        if self.underfill_started_at is None:
            self.underfill_started_at = timestamp
            return False
        if timestamp - self.underfill_started_at < self.critical_seconds:
            return False
        self.triggered = True
        return True

    def rates(self, *, now: Optional[float] = None) -> Dict[str, Any]:
        timestamp = time.monotonic() if now is None else float(now)
        for events in (
            self.new_events,
            self.proposal_events,
            self.repeat_events,
            self.invalid_events,
            self.cache_events,
            self.completion_events,
        ):
            self._prune(events, timestamp)
        elapsed = max(
            1.0,
            min(self.rate_window_seconds, timestamp - self.started_at),
        )
        supply = sum(count for _, count in self.new_events) / elapsed
        proposals = sum(count for _, count in self.proposal_events) / elapsed
        repeats = sum(count for _, count in self.repeat_events) / elapsed
        invalid = sum(count for _, count in self.invalid_events) / elapsed
        cache = sum(count for _, count in self.cache_events) / elapsed
        demand = sum(count for _, count in self.completion_events) / elapsed
        return {
            "recent_proposal_rate": float(proposals),
            "recent_candidate_supply_rate": float(supply),
            "recent_repeat_rate": float(repeats),
            "recent_invalid_rate": float(invalid),
            "recent_cache_rate": float(cache),
            "recent_new_per_proposal": float(
                supply / proposals if proposals > 0.0 else 0.0
            ),
            "recent_repeat_per_proposal": float(
                repeats / proposals if proposals > 0.0 else 0.0
            ),
            "recent_evaluator_completion_rate": float(demand),
            "recent_evaluator_demand_rate": float(demand),
            "recent_supply_ratio": (
                float(supply / demand) if demand > 0.0 else None
            ),
            "scheduler_supply_rate_window_seconds": float(elapsed),
        }
