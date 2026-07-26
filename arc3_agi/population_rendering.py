from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from arc3_agi.automaton import AutomatonBase


@dataclass(frozen=True)
class AutomatonGenerationSnapshot:
    fitness: float
    fingerprint_bits: int | None
    fingerprint_value: int | None


@dataclass(frozen=True)
class PopulationGenerationSnapshot:
    """Immutable evaluated-population data captured before evolution resets it."""

    automata: tuple[AutomatonGenerationSnapshot, ...]

    @classmethod
    def capture(cls, automata: Iterable[AutomatonBase]) -> PopulationGenerationSnapshot:
        return cls(
            tuple(
                AutomatonGenerationSnapshot(
                    fitness=float(automaton.fitness),
                    fingerprint_bits=(
                        automaton.fingerprint.bits
                        if automaton.fingerprint is not None
                        else None
                    ),
                    fingerprint_value=(
                        automaton.fingerprint.value
                        if automaton.fingerprint is not None
                        else None
                    ),
                )
                for automaton in automata
            )
        )

    @property
    def fitnesses(self) -> tuple[float, ...]:
        return tuple(automaton.fitness for automaton in self.automata)


def _moving_average(data: Sequence[float], window: int) -> list[float]:
    """Return a moving average, with unavailable positions filled by NaN."""
    result: list[float] = []
    for index in range(len(data)):
        if index + 1 < window:
            result.append(float("nan"))
        else:
            result.append(sum(data[index + 1 - window : index + 1]) / window)
    return result


def _install_webagg_disconnect_guard() -> None:
    """Make WebAgg tolerate clients disconnecting during a frame send."""
    if "webagg" not in plt.get_backend().lower():
        return

    from matplotlib.backends.backend_webagg import WebAggApplication
    from tornado.ioloop import IOLoop
    from tornado.iostream import StreamClosedError
    from tornado.websocket import WebSocketClosedError

    socket_class = WebAggApplication.WebSocket
    if getattr(socket_class, "_arc3_disconnect_guard", False):
        return

    class DisconnectSafeWebSocket(socket_class):
        _arc3_disconnect_guard = True

        def _discard(self) -> None:
            manager = getattr(self, "manager", None)
            if manager is not None:
                manager.web_sockets.discard(self)

        def on_close(self) -> None:
            self._discard()

        def _schedule_discard(self) -> None:
            IOLoop.current().add_callback(self._discard)

        def _finish_send(self, future) -> None:
            try:
                future.result()
            except (WebSocketClosedError, StreamClosedError):
                self._schedule_discard()

        def _send(self, payload, *, binary: bool = False) -> None:
            try:
                future = self.write_message(payload, binary=binary)
            except WebSocketClosedError:
                self._schedule_discard()
                return
            future.add_done_callback(self._finish_send)

        def send_json(self, content) -> None:
            self._send(json.dumps(content))

        def send_binary(self, blob) -> None:
            if self.supports_binary:
                self._send(blob, binary=True)
                return
            encoded = base64.b64encode(blob).decode("ascii")
            self._send(f"data:image/png;base64,{encoded}")

    WebAggApplication.WebSocket = DisconnectSafeWebSocket


class FitnessHistoryRenderer:
    """Live line chart showing mean and max fitness per generation."""

    def __init__(self, window: int = 10) -> None:
        self._window = window
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Fitness History")
        self._means: list[float] = []
        self._maxes: list[float] = []
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Fitness")
        self.ax.set_title("Fitness History")
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _redraw(self) -> None:
        generations = list(range(1, len(self._means) + 1))
        mean_average = _moving_average(self._means, self._window)
        max_average = _moving_average(self._maxes, self._window)
        self.ax.cla()
        self.ax.plot(
            generations,
            self._means,
            color="crimson",
            linewidth=1.5,
            label="mean",
        )
        self.ax.plot(
            generations,
            mean_average,
            color="crimson",
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
            label=f"mean MA-{self._window}",
        )
        self.ax.plot(
            generations,
            self._maxes,
            color="steelblue",
            linewidth=1.5,
            label="max",
        )
        self.ax.plot(
            generations,
            max_average,
            color="steelblue",
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
            label=f"max MA-{self._window}",
        )
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Fitness")
        self.ax.set_title("Fitness History")
        self.ax.legend()
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def update(self, fitnesses: Sequence[float], *, render: bool = True) -> None:
        self._means.append(sum(fitnesses) / len(fitnesses))
        self._maxes.append(max(fitnesses))
        if render:
            self._redraw()

    def render(self) -> None:
        if not self._means:
            return
        self._redraw()

    def is_open(self) -> bool:
        return plt.fignum_exists(self.fig.number)

    def close(self) -> None:
        plt.close(self.fig)


class FitnessRenderer:
    """Live histogram showing the fitness distribution of the population."""

    def __init__(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Fitness Distribution")
        self.generation = 0
        self._latest_fitnesses: Sequence[float] = ()
        self.ax.set_xlabel("Fitness")
        self.ax.set_ylabel("Count")
        self.ax.set_title("Generation 0 \u2013 Fitness Distribution")
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _redraw(self) -> None:
        if not self._latest_fitnesses:
            return
        fitnesses = self._latest_fitnesses
        self.ax.cla()
        self.ax.hist(fitnesses, bins=20, color="steelblue", edgecolor="black")
        minimum = min(fitnesses)
        maximum = max(fitnesses)
        mean = sum(fitnesses) / len(fitnesses)
        self.ax.axvline(
            mean,
            color="crimson",
            linestyle="--",
            linewidth=1.5,
            label=f"mean={mean:.1f}",
        )
        self.ax.set_xlabel("Fitness")
        self.ax.set_ylabel("Count")
        self.ax.set_title(
            f"Generation {self.generation}  \u2013  min={minimum:.1f}  "
            f"mean={mean:.1f}  max={maximum:.1f}"
        )
        self.ax.legend()
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def update(self, fitnesses: Sequence[float], *, render: bool = True) -> None:
        self.generation += 1
        self._latest_fitnesses = fitnesses
        if render:
            self._redraw()

    def render(self) -> None:
        self._redraw()

    def is_open(self) -> bool:
        return plt.fignum_exists(self.fig.number)

    def close(self) -> None:
        plt.close(self.fig)


class GenerationsPerSecondRenderer:
    """Live line chart showing generations per second over time."""

    def __init__(self, window: int = 10) -> None:
        self._window = window
        self._rates: list[float] = []
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Generations per Second")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Generations / s")
        self.ax.set_title("Generations per Second")
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _redraw(self) -> None:
        if not self._rates:
            return
        generations = list(range(1, len(self._rates) + 1))
        moving_average = _moving_average(self._rates, self._window)
        self.ax.cla()
        self.ax.plot(
            generations,
            self._rates,
            color="steelblue",
            linewidth=1.5,
            label="gens/s",
        )
        self.ax.plot(
            generations,
            moving_average,
            color="steelblue",
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
            label=f"MA-{self._window}",
        )
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Generations / s")
        self.ax.set_title("Generations per Second")
        self.ax.legend()
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def update(self, duration_s: float | None, *, render: bool = True) -> None:
        if duration_s is None or duration_s <= 0.0:
            return
        self._rates.append(1.0 / duration_s)
        if render:
            self._redraw()

    def render(self) -> None:
        self._redraw()

    def is_open(self) -> bool:
        return plt.fignum_exists(self.fig.number)

    def close(self) -> None:
        plt.close(self.fig)


class FitnessRateRenderer:
    """Live chart showing per-generation changes in max and mean fitness."""

    def __init__(self, window: int = 100) -> None:
        self._window = window
        self._prev_max: float | None = None
        self._prev_mean: float | None = None
        self._delta_maxes: list[float] = []
        self._delta_means: list[float] = []
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Fitness Rate of Change")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("\u0394 Fitness")
        self.ax.set_title("Fitness Rate of Change")
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _redraw(self) -> None:
        if not self._delta_maxes:
            return
        generations = list(range(1, len(self._delta_maxes) + 1))
        max_average = _moving_average(self._delta_maxes, self._window)
        mean_average = _moving_average(self._delta_means, self._window)
        self.ax.cla()
        self.ax.plot(
            generations,
            max_average,
            color="steelblue",
            linewidth=1.5,
            label=f"\u0394 max MA-{self._window}",
        )
        self.ax.plot(
            generations,
            mean_average,
            color="crimson",
            linewidth=1.5,
            label=f"\u0394 mean MA-{self._window}",
        )
        self.ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("\u0394 Fitness")
        self.ax.set_title("Fitness Rate of Change")
        self.ax.legend()
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def update(self, fitnesses: Sequence[float], *, render: bool = True) -> None:
        current_max = max(fitnesses)
        current_mean = sum(fitnesses) / len(fitnesses)
        if self._prev_max is None or self._prev_mean is None:
            self._delta_maxes.append(float("nan"))
            self._delta_means.append(float("nan"))
        else:
            self._delta_maxes.append(current_max - self._prev_max)
            self._delta_means.append(current_mean - self._prev_mean)
        self._prev_max = current_max
        self._prev_mean = current_mean
        if render:
            self._redraw()

    def render(self) -> None:
        self._redraw()

    def is_open(self) -> bool:
        return plt.fignum_exists(self.fig.number)

    def close(self) -> None:
        plt.close(self.fig)


class FingerprintClusterRenderer:
    """Live 2-D PCA scatter plot of selection-fingerprint bit vectors."""

    def __init__(self) -> None:
        self._generation = 0
        self._cbar = None
        self._latest_snapshot: PopulationGenerationSnapshot | None = None
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Fingerprint Clusters")
        self.ax.set_xlabel("PC 1")
        self.ax.set_ylabel("PC 2")
        self.ax.set_title("Generation 0 \u2013 Fingerprint Clusters")
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _redraw(self) -> None:
        snapshot = self._latest_snapshot
        if snapshot is None:
            return
        members = [
            automaton
            for automaton in snapshot.automata
            if automaton.fingerprint_bits is not None
            and automaton.fingerprint_value is not None
        ]
        if not members:
            return

        bits = members[0].fingerprint_bits
        assert bits is not None
        count = len(members)
        values = np.array(
            [automaton.fingerprint_value for automaton in members], dtype=np.int64
        )
        fitnesses = np.array(
            [automaton.fitness for automaton in members], dtype=np.float32
        )
        bit_positions = np.arange(bits, dtype=np.int64)
        matrix = ((values[:, None] >> bit_positions[None, :]) & 1).astype(np.float32)

        centered = matrix - matrix.mean(axis=0)
        if centered.any() and count >= 2:
            try:
                _, _, components = np.linalg.svd(centered, full_matrices=False)
                component_count = min(2, components.shape[0])
                coordinates = centered @ components[:component_count].T
                if coordinates.shape[1] < 2:
                    coordinates = np.column_stack(
                        [coordinates, np.zeros(count, dtype=np.float32)]
                    )
            except np.linalg.LinAlgError:
                coordinates = np.zeros((count, 2), dtype=np.float32)
        else:
            coordinates = np.zeros((count, 2), dtype=np.float32)

        unique_values, inverse = np.unique(values, return_inverse=True)
        unique_count = len(unique_values)
        aggregate_x = np.zeros(unique_count, dtype=np.float32)
        aggregate_y = np.zeros(unique_count, dtype=np.float32)
        aggregate_fitness = np.zeros(unique_count, dtype=np.float32)
        aggregate_count = np.zeros(unique_count, dtype=np.int32)
        np.add.at(aggregate_x, inverse, coordinates[:, 0])
        np.add.at(aggregate_y, inverse, coordinates[:, 1])
        np.add.at(aggregate_fitness, inverse, fitnesses)
        np.add.at(aggregate_count, inverse, 1)
        aggregate_x /= aggregate_count
        aggregate_y /= aggregate_count
        aggregate_fitness /= aggregate_count

        largest_group = int(aggregate_count.max())
        sizes = 30 + 570 * (aggregate_count / max(largest_group, 1))
        minimum_fitness = float(fitnesses.min())
        maximum_fitness = float(fitnesses.max())
        fitness_range = (
            maximum_fitness - minimum_fitness
            if maximum_fitness > minimum_fitness
            else 1.0
        )

        if self._cbar is not None:
            self._cbar.remove()
            self._cbar = None

        self.ax.cla()
        scatter = self.ax.scatter(
            aggregate_x,
            aggregate_y,
            c=aggregate_fitness,
            s=sizes,
            cmap="plasma",
            alpha=0.85,
            vmin=minimum_fitness,
            vmax=minimum_fitness + fitness_range,
            linewidths=0.5,
            edgecolors="gray",
        )
        for index in range(unique_count):
            self.ax.annotate(
                str(aggregate_count[index]),
                (float(aggregate_x[index]), float(aggregate_y[index])),
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
            )
        self._cbar = self.fig.colorbar(scatter, ax=self.ax, label="Mean Fitness")
        self.ax.set_xlabel("PC 1")
        self.ax.set_ylabel("PC 2")
        self.ax.set_title(
            f"Generation {self._generation} \u2013 Fingerprint Clusters\n"
            f"{unique_count}/{1 << bits} unique fingerprints  "
            "(area \u221d count)"
        )
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def update(
        self, snapshot: PopulationGenerationSnapshot, *, render: bool = True
    ) -> None:
        members = [
            automaton
            for automaton in snapshot.automata
            if automaton.fingerprint_bits is not None
            and automaton.fingerprint_value is not None
        ]
        if not members:
            return
        self._generation += 1
        self._latest_snapshot = snapshot
        if render:
            self._redraw()

    def render(self) -> None:
        self._redraw()

    def is_open(self) -> bool:
        return plt.fignum_exists(self.fig.number)

    def close(self) -> None:
        plt.close(self.fig)


class PopulationChartSuite:
    """Own and update the population-level charts used by interactive runs."""

    def __init__(self) -> None:
        _install_webagg_disconnect_guard()
        self.fitness = FitnessRenderer()
        self.fitness_history = FitnessHistoryRenderer()
        self.generations_per_second = GenerationsPerSecondRenderer()
        self.fitness_rate = FitnessRateRenderer()
        self.fingerprint_clusters = FingerprintClusterRenderer()
        self._renderers = (
            self.fitness,
            self.fitness_history,
            self.generations_per_second,
            self.fitness_rate,
            self.fingerprint_clusters,
        )

    @property
    def canvas(self):
        return self.fitness.fig.canvas

    @property
    def figures(self) -> tuple:
        return tuple(renderer.fig for renderer in self._renderers)

    def update(
        self,
        snapshot: PopulationGenerationSnapshot,
        duration_s: float | None,
        *,
        render: bool = True,
    ) -> None:
        fitnesses = snapshot.fitnesses
        if not fitnesses:
            return
        self.fitness.update(fitnesses, render=render)
        self.fitness_history.update(fitnesses, render=render)
        self.generations_per_second.update(duration_s, render=render)
        self.fitness_rate.update(fitnesses, render=render)
        self.fingerprint_clusters.update(snapshot, render=render)

    def render(self) -> None:
        self.fitness.render()
        self.fitness_history.render()
        self.generations_per_second.render()
        self.fitness_rate.render()
        self.fingerprint_clusters.render()

    def update_batch(
        self,
        snapshots: Sequence[PopulationGenerationSnapshot],
        durations_s: Sequence[float | None],
    ) -> None:
        for snapshot, duration_s in zip(snapshots, durations_s):
            self.update(snapshot, duration_s, render=False)
        self.render()

    def is_open(self) -> bool:
        return any(renderer.is_open() for renderer in self._renderers)

    def close(self) -> None:
        for renderer in self._renderers:
            renderer.close()
