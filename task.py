#!/usr/bin/env python3
"""Editable coursework sequence. ./task.py runs; ./task.py --validate checks only.

Durations are nominal simulation seconds. Planning and measured settling add time.
robot.delay() is a guarded simulated hold, not an unmonitored wall-clock sleep.
"""
from pathlib import Path
import sys

# The task remains runnable from any directory, including Python safe-path mode.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gp7_demo import TIMING_PROFILES, run

PROFILE = "fast"
TIMINGS = dict(TIMING_PROFILES[PROFILE]["timings"])
# Override individual durations here, for example: TIMINGS["transfer"] = 4.5.
# None for "approach" retains the existing automatically selected safe duration.

PICKUP_APPROACH_HEIGHT_M = 0.25
CARRY_CLEARANCE_M = 0.05
RETREAT_DISTANCE_M = 0.10


def pick_and_place(robot):
    robot.raise_if_needed()
    robot.delay(0.5)

    robot.approach_pickup()
    robot.delay(0.5)

    robot.descend_to_cube()
    robot.delay(0.5)

    robot.grasp()
    robot.delay(1.0)

    robot.lift()
    robot.delay(1.0)
    if robot.stop_after_lift:
        return

    robot.transfer_polar()
    robot.delay(1.0)

    robot.lower_onto_pedestal()
    robot.delay(1.0)

    robot.release()
    robot.delay(0.5)

    robot.retreat()
    robot.verify_placement()


if __name__ == "__main__":
    raise SystemExit(run(
        pick_and_place,
        timings=TIMINGS,
        profile=PROFILE,
        pickup_height=PICKUP_APPROACH_HEIGHT_M,
        hover_clearance=CARRY_CLEARANCE_M,
        retreat_distance=RETREAT_DISTANCE_M,
    ))
