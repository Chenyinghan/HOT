"""Milestone observations and deduplicated diagnostic reward counts."""
from dataclasses import dataclass, field
import bisect
import threading
from typing import Any, Dict, List, Optional, Tuple

@dataclass(frozen=True)
class StageEvidence:
    """One ordered task-stage observation attached to a terminal reward."""

    milestone: int
    stage_count: int
    progress: float
    task_success: bool
    feasible: bool = True
    residual_reward: float = 0.0

@dataclass
class RewardStatistics:
    """Numerically stable reward evidence for one completion set."""

    count: int = 0
    total: float = 0.0
    maximum: float = float("-inf")
    samples: List[float] = field(default_factory=list)
    stage_samples: List[StageEvidence] = field(default_factory=list)
    terminal_count: int = 0
    _terminal_keys: set = field(default_factory=set, repr=False)
    _value_cache: Dict[Tuple[Any, ...], float] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _structured_value_cache: Dict[
        Tuple[Any, ...], Tuple[float, Dict[str, Any]]
    ] = field(default_factory=dict, repr=False, compare=False)
    _lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
        compare=False,
    )

    @property
    def mean(self) -> float:
        with self._lock:
            return self.total / float(self.count) if self.count else 0.0

    def record(
        self,
        reward: float,
        terminal_key: Optional[Any] = None,
        *,
        store_sample: bool = True,
        stage_evidence: Optional[StageEvidence] = None,
    ) -> bool:
        with self._lock:
            if terminal_key is not None:
                if terminal_key in self._terminal_keys:
                    return False
                self._terminal_keys.add(terminal_key)
                self.terminal_count += 1
            self.count += 1
            self.total += float(reward)
            self.maximum = max(self.maximum, float(reward))
            if store_sample:
                bisect.insort(self.samples, float(reward))
            if stage_evidence is not None:
                self.stage_samples.append(stage_evidence)
            self._value_cache.clear()
            self._structured_value_cache.clear()
            return True
