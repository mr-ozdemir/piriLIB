"""From a forecast to a planning problem.

A plan made for the expected demand runs short on about half of the days. A
*buffer* raises the forecast of each line to a quantile of what actually came
in the backtest: with ``quantile=0.8`` the planned quantity covered the actual
one in about 80 % of the past forecasts of that line.

With :class:`~pirilib.forecast.WaveShares`, every day of a line is split into
time waves, and each wave becomes a demand row whose ``ready`` time is the
wave time.
"""

import datetime as dt
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pirilib.forecast.evaluate import Backtest
from pirilib.forecast.models import Forecast
from pirilib.forecast.waves import WaveShares

MIN_CELLS = 8


def buffer_factors(backtest: Backtest, quantile: float, lines: int) -> NDArray[np.float64]:
    """(lines,) factors that lift each line's forecast to ``quantile`` of actual / forecast.

    Lines with fewer than :data:`MIN_CELLS` scored cells get the factor of all
    lines together. A model that forecasts too high can get factors below 1.
    """
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")
    line = np.concatenate([fold.lines for fold in backtest.folds])
    actual = np.concatenate([fold.actual for fold in backtest.folds])
    predicted = np.concatenate([fold.predicted for fold in backtest.folds])
    usable = predicted > 0
    ratio = actual[usable] / predicted[usable]
    line = line[usable]
    overall = float(np.quantile(ratio, quantile)) if ratio.size else 1.0
    factors = np.full(lines, overall)
    for index in range(lines):
        own = ratio[line == index]
        if own.size >= MIN_CELLS:
            factors[index] = float(np.quantile(own, quantile))
    return factors


def demand_rows(
    forecast: Forecast,
    *,
    factors: NDArray[np.float64] | None = None,
    minimum: float = 0.5,
    waves: WaveShares | None = None,
) -> list[dict[str, Any]]:
    """Demand rows of a problem document, one per line and day of at least ``minimum``.

    With ``waves``, one row per line, day and wave of at least ``minimum``,
    ready at the wave time; pieces below ``minimum`` are dropped.
    """
    values = forecast.values if factors is None else forecast.values * factors[:, None]
    lifted = Forecast(
        forecast.origins, forecast.destinations, forecast.start, values, forecast.dimension
    )
    if waves is not None:
        return [
            {
                "origin": origin,
                "destination": destination,
                "date": date.isoformat(),
                "ready": ready.isoformat("minutes"),
                "quantity": {forecast.dimension: round(quantity, 3)},
            }
            for origin, destination, date, ready, quantity in waves.split(lifted).rows(minimum)
        ]
    return [
        {
            "origin": origin,
            "destination": destination,
            "date": date.isoformat(),
            "quantity": {forecast.dimension: round(quantity, 3)},
        }
        for origin, destination, date, quantity in lifted.rows(minimum)
    ]


def planning_document(
    network: Mapping[str, Any],
    forecast: Forecast,
    *,
    factors: NDArray[np.float64] | None = None,
    minimum: float = 0.5,
    waves: WaveShares | None = None,
) -> dict[str, Any]:
    """A problem document: the network (centers, fleet, rules ...) with the forecast as demand.

    The calendar becomes the forecast window. With ``waves``, the demand of
    every line and day is split into time waves that are ready at the wave
    time (see :func:`demand_rows`); ``rules.service.due_hours`` of the network
    then counts from that time. Validate it with
    :func:`pirilib.spec.parse_spec`; lines whose centers are not in the network
    are reported there.
    """
    end = forecast.start + dt.timedelta(days=forecast.days - 1)
    return {
        **network,
        "calendar": {"start": forecast.start.isoformat(), "end": end.isoformat()},
        "demand": demand_rows(forecast, factors=factors, minimum=minimum, waves=waves),
    }
