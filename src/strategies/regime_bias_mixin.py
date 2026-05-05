from typing import Optional
from .base_strategy import Signal, SignalType


class RegimeBiasMixin:
    """Suppresses counter-trend entry signals based on _regime_hint injected by orchestrator.

    Only suppresses LONG/SHORT entries. Close signals always pass through
    so open positions are never stranded by the bias filter.
    """

    def _apply_regime_bias(self, signal: Optional[Signal]) -> Optional[Signal]:
        if signal is None:
            return None
        # Only filter entry signals — never filter exits
        if signal.signal_type not in (SignalType.LONG, SignalType.SHORT):
            return signal
        hint: str = (
            getattr(self, "config", None)
            and self.config.params.get("_regime_hint", "unknown")
            or "unknown"
        )
        if hint == "trending_up" and signal.signal_type == SignalType.SHORT:
            return None
        if hint == "trending_down" and signal.signal_type == SignalType.LONG:
            return None
        return signal
