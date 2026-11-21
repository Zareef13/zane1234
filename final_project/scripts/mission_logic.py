#!/usr/bin/env python3
"""Pure decision logic for the Saving Christmas mission.

The ROS node in mission_supervisor.py feeds this class detections and the
robot pose. Keeping the rules here makes them testable without a robot.
"""

import math


EXPLORE = "explore"
STOP = "stop"
CELEBRATE = "celebrate"

STOP_SIGN = "stop_sign"
PERSON = "person"
REINDEER = "reindeer"
CLASSES = (STOP_SIGN, PERSON, REINDEER)


def classify_label(text):
    """Map a detector string onto stop_sign, person, reindeer, or None."""
    if text is None:
        return None
    name = str(text).strip().lower().replace("-", " ").replace("_", " ")
    name = " ".join(name.split())
    if name in {"stop sign", "stopsign", "stop"}:
        return STOP_SIGN
    if name in {"person", "pedestrian", "people", "human"}:
        return PERSON
    if name in {"teddy bear", "teddybear", "reindeer", "rudolph", "deer"}:
        return REINDEER
    return None


class DetectionLatch:
    """Turn noisy detector messages into a stable yes or no per class.

    A class is confirmed only after `confirm_hits` messages arrive with no
    gap longer than `timeout`. One stray frame does not change the mission.
    """

    def __init__(self, confirm_hits=2, timeout=0.75):
        self.confirm_hits = confirm_hits
        self.timeout = timeout
        self.hits = {kind: 0 for kind in CLASSES}
        self.last_hit = {kind: None for kind in CLASSES}

    def mark(self, kind, now):
        last = self.last_hit[kind]
        if last is None or now - last > self.timeout:
            self.hits[kind] = 1
        else:
            self.hits[kind] += 1
        self.last_hit[kind] = now

    def clear(self, kind):
        self.hits[kind] = 0
        self.last_hit[kind] = None

    def confirmed(self, kind, now):
        last = self.last_hit[kind]
        if last is None or now - last > self.timeout:
            self.hits[kind] = 0
            return False
        return self.hits[kind] >= self.confirm_hits


class MissionLogic:
    """Explore, obey the two traffic rules, then celebrate finding Rudolph.

    Priority is reindeer, then pedestrian, then stop sign. A stop sign is
    held for `stop_duration` seconds and is not obeyed again until the robot
    has driven `stop_sign_clear_distance` meters away. A pedestrian holds the
    robot until detections have been gone for `pedestrian_clear_time`.
    """

    def __init__(
        self,
        stop_duration=5.0,
        pedestrian_clear_time=1.0,
        stop_sign_clear_distance=0.8,
    ):
        self.stop_duration = stop_duration
        self.pedestrian_clear_time = pedestrian_clear_time
        self.stop_sign_clear_distance = stop_sign_clear_distance

        self.mode = EXPLORE
        self.event = None
        self.holding_for_sign = False
        self.holding_for_person = False
        self.hold_until = None
        self.sign_anchor = None
        self.person_clear_since = None
        self.found = False

    def update(self, now, x, y, sees_sign, sees_person, sees_reindeer):
        self.event = None

        if self.found:
            self.mode = CELEBRATE
            return self.mode

        if sees_reindeer:
            self.found = True
            self.event = REINDEER
            self.mode = CELEBRATE
            return self.mode

        if self.mode == STOP:
            self._update_stop(now, x, y, sees_sign, sees_person)
            if self.holding_for_sign or self.holding_for_person:
                self.mode = STOP
                return self.mode
            self.mode = EXPLORE
            self.event = "resume"

        sign_blocked = self._sign_blocked(x, y)
        if sees_person or (sees_sign and not sign_blocked):
            self._begin_stop(
                now,
                for_person=sees_person,
                for_sign=sees_sign and not sign_blocked,
            )
            return self.mode

        self.mode = EXPLORE
        return self.mode

    def _begin_stop(self, now, for_person, for_sign):
        self.mode = STOP
        if for_sign and not self.holding_for_sign:
            self.holding_for_sign = True
            self.hold_until = now + self.stop_duration
            self.event = STOP_SIGN
        if for_person and not self.holding_for_person:
            self.holding_for_person = True
            self.person_clear_since = None
            self.event = PERSON

    def _update_stop(self, now, x, y, sees_sign, sees_person):
        if sees_person:
            if not self.holding_for_person:
                self.holding_for_person = True
                self.event = PERSON
            self.person_clear_since = None
        elif self.holding_for_person:
            if self.person_clear_since is None:
                self.person_clear_since = now
            elif now - self.person_clear_since >= self.pedestrian_clear_time:
                self.holding_for_person = False
                self.person_clear_since = None

        if sees_sign and not self.holding_for_sign and not self._sign_blocked(x, y):
            self.holding_for_sign = True
            self.hold_until = now + self.stop_duration
            if self.event is None:
                self.event = STOP_SIGN

        if self.holding_for_sign and self.hold_until is not None and now >= self.hold_until:
            self.holding_for_sign = False
            self.hold_until = None
            self.sign_anchor = (x, y)

    def _sign_blocked(self, x, y):
        if self.sign_anchor is None:
            return False
        distance = math.hypot(x - self.sign_anchor[0], y - self.sign_anchor[1])
        if distance >= self.stop_sign_clear_distance:
            self.sign_anchor = None
            return False
        return True


def _check(name, condition):
    if not condition:
        raise AssertionError(name)


def _run_tests():
    logic = MissionLogic(stop_duration=5.0, pedestrian_clear_time=1.0, stop_sign_clear_distance=0.8)
    assert logic.update(0.0, 0.0, 0.0, False, False, False) == EXPLORE

    assert logic.update(0.1, 0.0, 0.0, True, False, False) == STOP
    assert logic.event == STOP_SIGN
    assert logic.update(5.0, 0.0, 0.0, True, False, False) == STOP
    assert logic.update(5.1, 0.0, 0.0, True, False, False) == EXPLORE
    assert logic.update(5.2, 0.0, 0.0, True, False, False) == EXPLORE

    assert logic.update(6.0, 1.0, 0.0, True, False, False) == STOP
    assert logic.update(11.0, 1.0, 0.0, False, False, False) == EXPLORE

    assert logic.update(12.0, 2.0, 0.0, False, True, False) == STOP
    assert logic.event == PERSON
    assert logic.update(20.0, 2.0, 0.0, False, True, False) == STOP
    assert logic.update(20.2, 2.0, 0.0, False, False, False) == STOP
    assert logic.update(21.2, 2.0, 0.0, False, False, False) == EXPLORE

    assert logic.update(22.0, 3.0, 0.0, True, True, False) == STOP
    assert logic.event == PERSON
    assert logic.update(27.0, 3.0, 0.0, False, True, False) == STOP
    assert logic.update(27.5, 3.0, 0.0, False, False, False) == STOP
    assert logic.update(28.5, 3.0, 0.0, False, False, False) == EXPLORE

    both = MissionLogic()
    assert both.update(0.0, 0.0, 0.0, True, False, False) == STOP
    assert both.update(1.0, 0.0, 0.0, True, True, False) == STOP
    assert both.event == PERSON
    assert both.update(6.0, 0.0, 0.0, False, True, False) == STOP
    assert both.update(6.0, 0.0, 0.0, False, False, False) == STOP
    assert both.update(7.0, 0.0, 0.0, False, False, False) == EXPLORE

    assert logic.update(30.0, 4.0, 0.0, True, False, True) == CELEBRATE
    assert logic.event == REINDEER
    assert logic.update(40.0, 4.0, 0.0, True, True, False) == CELEBRATE

    latch = DetectionLatch(confirm_hits=2, timeout=0.75)
    latch.mark(STOP_SIGN, 0.0)
    _check("one hit is not enough", not latch.confirmed(STOP_SIGN, 0.1))
    latch.mark(STOP_SIGN, 0.2)
    _check("two hits confirm", latch.confirmed(STOP_SIGN, 0.3))
    _check("stale hit drops", not latch.confirmed(STOP_SIGN, 1.2))

    _check("stop sign label", classify_label("Stop Sign") == STOP_SIGN)
    _check("person label", classify_label("pedestrian") == PERSON)
    _check("teddy bear is rudolph", classify_label("teddy_bear") == REINDEER)
    _check("unknown label", classify_label("potted plant") is None)
    print("mission_logic tests passed")


if __name__ == "__main__":
    _run_tests()
