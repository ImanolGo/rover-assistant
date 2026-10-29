"""brain.mission: the mission loop (PLAN 5.5, ARCHITECTURE §4).

``PLAN -> (SEARCH | APPROACH | REACQUIRE | ...) -> VERIFY -> DONE | REPLAN``, one
mission at a time on its own thread. Each tick builds a :class:`WorldState` from
the newest detections, asks the selector for a skill, runs it, and logs the
decision. The mission drives through a *body* (the app: ``snapshot``, ``drive``,
``halt``, ``heading_deg``, ``say``, ``set_state``), so barge-in stop follows the
wheels and a spoken stop cancels it via :meth:`Mission.cancel`.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from rover.brain import skills
from rover.brain.state import Skill, WorldState
from rover.brain.verify import verify_arrival
from rover.perception.color import COLOR_RANGES, matches_color

HOLD_S = 0.5  # keep the last detection this long: YOLO flickered a real cup on/off
# every other tick, and each "lost" halted the wheels (robot run 2026-10-29)


class _Motor:
    """What :mod:`skills` drives: the body's wheels, with ``stop`` = halt (not abort).

    ``drive`` checks the cancel flag under the mission lock, and :meth:`Mission.cancel`
    sets it under the same lock, so no drive can slip in after a stop.
    """

    def __init__(self, mission: "Mission", body: Any):
        self.mission, self.stop, self.heading_deg = mission, body.halt, body.heading_deg
        self._drive = body.drive

    def drive(self, left: float, right: float) -> None:
        with self.mission._lock:
            if not self.mission.cancelled():
                self._drive(left, right)


class Mission:
    def __init__(self, body, gemma, selector, log, control, spin_speed, half_fov_deg, h_stops):
        self.body, self.gemma, self.selector, self.log = body, gemma, selector, log
        self.control, self.spin_speed, self.half_fov = control, spin_speed, half_fov_deg
        self.h_stops = dict(h_stops or {})
        self._lock = threading.Lock()
        self.motor = _Motor(self, body)
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    # --- lifecycle -------------------------------------------------------------

    def start(self, run: Callable[..., None], *args: Any) -> None:
        """Cancel any running mission, then run ``run(*args)`` on a fresh thread."""
        self.cancel()
        self._cancel = threading.Event()
        self._thread = threading.Thread(target=self._guard, args=(run, args), daemon=True)
        self._thread.start()

    def _guard(self, run: Callable[..., None], args: tuple) -> None:
        try:
            run(*args)
        except Exception as exc:  # noqa: BLE001 - a mission must never kill the brain
            self.body.halt()
            print(f"mission failed: {type(exc).__name__}: {exc}", flush=True)
            self.body.say("Sorry, something went wrong.")
        finally:
            if not self.cancelled():
                self.body.set_state("IDLE")

    def cancel(self, join: bool = True) -> None:
        with self._lock:
            self._cancel.set()
        thread = self._thread
        if join and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def wait(self, seconds: float) -> None:
        self._cancel.wait(seconds)

    # --- missions --------------------------------------------------------------

    def go_to(self, target: str, attributes: tuple[str, ...] = ()) -> None:
        self._seek(WorldState(mission="go_to", target=target, attributes=tuple(attributes)))

    def follow(self) -> None:
        self._seek(WorldState(mission="follow", target="person"))

    def describe(self, question: str) -> None:
        self.body.halt()
        self.body.say("Let me look.")
        snap = self.body.snapshot()
        labels = ", ".join(sorted({d.cls for d in snap.detections})) or "nothing"
        prompt = f"{question}\nThe detector sees: {labels}. Answer in one or two short sentences."
        self.body.say(self.ask(snap.frame, prompt) or "Sorry, I can't tell right now.")

    def ask(self, frame: Any, prompt: str) -> str:
        """Gemma vision question; "" when llama-server is down or errors (soft failure)."""
        try:
            return self.gemma.ask_image(frame, prompt)
        except Exception as exc:  # noqa: BLE001 - a dead planner must not end the mission
            print(f"gemma unavailable: {type(exc).__name__}: {exc}", flush=True)
            return ""

    def turn(self, degrees: float) -> None:
        skills.spin(self.motor, degrees, self.spin_speed, self.cancelled)

    def nudge(self, speed: float, seconds: float = 1.0) -> None:
        """Drive straight (``speed`` < 0: back) for ``seconds``, then halt."""
        end = time.monotonic() + seconds
        while time.monotonic() < end and not self.cancelled():
            self.motor.drive(speed, speed)
            self.wait(0.05)
        self.body.halt()

    def _seek(self, s: WorldState) -> None:
        goal = " ".join([*s.attributes, s.target or ""]).strip()
        s.h_stop = float(self.h_stops.get(s.target, s.h_stop))
        if s.mission == "go_to":
            self.body.say(f"Looking for the {goal}.")
        started, last_seen, last_hit = time.monotonic(), None, None
        while not self.cancelled():
            snap = self.body.snapshot()
            hit = self._match(snap, s.target, s.attributes)
            now = time.monotonic()
            if hit is not None:
                last_hit, last_seen = hit, now
            elif last_seen is not None and now - last_seen <= HOLD_S:
                hit = last_hit  # a dropout, not a loss
            s.elapsed_s = now - started
            s.target_visible = hit is not None
            if hit is not None:
                s.lost_for_s = 0.0
                s.bearing_deg = s.last_bearing_deg = hit.bearing_deg
                s.h_frac = hit.h_frac
            else:
                s.bearing_deg = s.h_frac = None
                s.lost_for_s = now - (last_seen if last_seen is not None else started)
            skill = self.selector.select(s)
            self.body.set_state(skill.value)
            outcome = self._run(skill, s, snap, hit, goal)
            self.log.append(s, skill, outcome)
            s.last_skill = skill
            if skill in (Skill.DONE, Skill.ABORT):
                return
            self.wait(1.0 / self.control.tick_hz)
        self.body.halt()

    def _match(self, snap: Any, target: str | None, attributes: tuple[str, ...]) -> Any:
        colours = [a for a in attributes if a in COLOR_RANGES]
        hits = [d for d in snap.detections if d.cls == target]
        if colours and snap.frame is not None:
            hits = [d for d in hits if all(matches_color(snap.frame, d.box, c) for c in colours)]
        return max(hits, key=lambda d: d.h_frac, default=None)

    def _run(self, skill: Skill, s: WorldState, snap: Any, hit: Any, goal: str) -> str:
        c = self.control
        if skill is Skill.SEARCH:
            turned = skills.spin(self.motor, c.search_step_deg, self.spin_speed, self.cancelled)
            s.search_steps += 1
            self.wait(c.search_pause_s)
            return f"turned {turned:.0f}"
        if skill is Skill.APPROACH:
            command = skills.approach(s.bearing_deg, s.h_frac, s.h_stop, self.half_fov, c)
            if command is None:  # far off-centre: re-aim in place first
                return f"re-aim {self.turn_toward(s.bearing_deg):.0f}"
            self.motor.drive(*command)
            return f"drive {command[0]:+.2f} {command[1]:+.2f}"
        if skill is Skill.FOLLOW:
            command = skills.follow(s.bearing_deg, s.h_frac, self.half_fov, c)
            if command is None:  # in range but off to the side: re-aim in place
                return f"re-aim {self.turn_toward(s.bearing_deg):.0f}"
            self.motor.drive(*command)
            return f"drive {command[0]:+.2f} {command[1]:+.2f}"
        if skill is Skill.REACQUIRE:
            self.body.halt()
            if s.lost_for_s <= c.lost_reacquire_s:
                return "hold"
            step = c.search_step_deg if (s.last_bearing_deg or 0.0) >= 0 else -c.search_step_deg
            return f"turned {self.turn_toward(step):.0f}"
        if skill is Skill.VERIFY:
            return self._verify(s, goal)
        if skill is Skill.BACK_OFF:
            skills.back_off(self.motor, c.v_max, self.cancelled)
            s.replans, s.verify_result, s.search_steps = s.replans + 1, None, 0
            self.body.say("That's not it. Looking again.")
            return "backed off"
        if skill is Skill.ASK_PLANNER:
            s.asked_planner = True
            answer = self.ask(snap.frame, f"Is there a {goal} here? Yes or no.")
            return f"gemma: {answer[:40]}"
        self.body.halt()
        found = skill is Skill.DONE and s.mission == "go_to"
        lost = "I lost you." if s.mission == "follow" else f"I couldn't find the {goal}."
        self.body.say(f"I'm at the {goal}." if found else lost)
        return skill.value.lower()

    def turn_toward(self, bearing_deg: float) -> float:
        """Spin so a target at ``bearing_deg`` (+ = right) comes to the centre."""
        return skills.spin(self.motor, -bearing_deg, self.spin_speed, self.cancelled)

    def _verify(self, s: WorldState, goal: str) -> str:
        self.body.halt()
        confidences, snap, hit = [], None, None
        for _ in range(self.control.verify_frames):
            self.wait(0.15)
            snap = self.body.snapshot()
            hit = self._match(snap, s.target, s.attributes)
            confidences.append(hit.conf if hit is not None else None)
        colours = [a for a in s.attributes if a in COLOR_RANGES]
        frame = snap.frame if snap is not None else None
        colour_ok = hit is not None and (
            not colours
            or (frame is not None and all(matches_color(frame, hit.box, k) for k in colours))
        )
        arrived, reason = verify_arrival(
            confidences,
            colour_ok,
            lambda prompt: self.ask(frame, prompt),
            goal,
            self.control.verify_min_conf,
        )
        s.verify_result = arrived
        return f"verify {reason} {confidences}"
