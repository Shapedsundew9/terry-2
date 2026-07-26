import math
from random import Random
from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use("Agg")

from arc3_agi.fingerprint import SelectionFingerprint
from arc3_agi.population_rendering import (
    FingerprintClusterRenderer,
    PopulationChartSuite,
    PopulationGenerationSnapshot,
    _install_webagg_disconnect_guard,
    _moving_average,
)


def _automaton(fitness: float, fingerprint_value: int | None = None):
    fingerprint = (
        SelectionFingerprint(4, Random(0), value=fingerprint_value)
        if fingerprint_value is not None
        else None
    )
    return SimpleNamespace(fitness=fitness, fingerprint=fingerprint)


def test_moving_average_marks_incomplete_windows() -> None:
    result = _moving_average([1.0, 2.0, 3.0, 4.0], window=3)

    assert math.isnan(result[0])
    assert math.isnan(result[1])
    assert result[2:] == [2.0, 3.0]


def test_webagg_disconnect_guard_consumes_closed_send_futures(monkeypatch) -> None:
    from matplotlib.backends.backend_webagg import WebAggApplication
    from tornado.ioloop import IOLoop
    from tornado.websocket import WebSocketClosedError

    original_socket_class = WebAggApplication.WebSocket
    monkeypatch.setattr(WebAggApplication, "WebSocket", original_socket_class)
    monkeypatch.setattr(matplotlib.pyplot, "get_backend", lambda: "webagg")
    _install_webagg_disconnect_guard()
    socket_class = WebAggApplication.WebSocket

    class Manager:
        def __init__(self) -> None:
            self.web_sockets = set()

    class ClosedFuture:
        def add_done_callback(self, callback) -> None:
            callback(self)

        def result(self) -> None:
            raise WebSocketClosedError()

    class DeferredCallbacks:
        def __init__(self) -> None:
            self.callbacks = []

        def add_callback(self, callback) -> None:
            self.callbacks.append(callback)

        def run(self) -> None:
            for callback in self.callbacks:
                callback()
            self.callbacks.clear()

    callbacks = DeferredCallbacks()
    monkeypatch.setattr(IOLoop, "current", staticmethod(lambda: callbacks))

    manager = Manager()
    synchronous = object.__new__(socket_class)
    synchronous.manager = manager
    synchronous.supports_binary = True
    manager.web_sockets.add(synchronous)
    synchronous.write_message = lambda *args, **kwargs: (_ for _ in ()).throw(
        WebSocketClosedError()
    )

    synchronous.send_json({"type": "refresh"})

    assert synchronous in manager.web_sockets
    callbacks.run()
    assert synchronous not in manager.web_sockets

    asynchronous = object.__new__(socket_class)
    asynchronous.manager = manager
    asynchronous.supports_binary = True
    manager.web_sockets.add(asynchronous)
    asynchronous.write_message = lambda *args, **kwargs: ClosedFuture()

    asynchronous.send_binary(b"frame")

    assert asynchronous in manager.web_sockets
    callbacks.run()
    assert asynchronous not in manager.web_sockets


def test_snapshot_is_immutable_copy_of_evaluated_population() -> None:
    automata = [_automaton(1.0, 3), _automaton(2.0)]
    snapshot = PopulationGenerationSnapshot.capture(automata)

    automata[0].fitness = 99.0
    automata[0].fingerprint.value = 8

    assert snapshot.fitnesses == (1.0, 2.0)
    assert snapshot.automata[0].fingerprint_bits == 4
    assert snapshot.automata[0].fingerprint_value == 3
    assert snapshot.automata[1].fingerprint_value is None


def test_fingerprint_renderer_keeps_fitness_aligned_with_fingerprint() -> None:
    snapshot = PopulationGenerationSnapshot.capture(
        [_automaton(1.0, 0), _automaton(3.0, 0), _automaton(10.0, 1)]
    )
    renderer = FingerprintClusterRenderer()

    try:
        renderer.update(snapshot)
        aggregate_fitness = renderer.ax.collections[0].get_array()
        assert aggregate_fitness is not None
        assert np.allclose(np.sort(aggregate_fitness), [2.0, 10.0])
    finally:
        renderer.close()


def test_chart_suite_updates_all_charts_and_closes_figures() -> None:
    snapshot = PopulationGenerationSnapshot.capture(
        [_automaton(0.25, 1), _automaton(0.75, 2)]
    )
    charts = PopulationChartSuite()

    charts.update(snapshot, duration_s=0.5)

    assert charts.fitness.generation == 1
    assert charts.fitness_history._means == [0.5]
    assert charts.fitness_history._maxes == [0.75]
    assert charts.generations_per_second._rates == [2.0]
    assert len(charts.fitness_rate._delta_means) == 1
    assert charts.fingerprint_clusters._generation == 1
    assert len(charts.figures) == 5
    assert charts.is_open()

    charts.close()

    assert not charts.is_open()
