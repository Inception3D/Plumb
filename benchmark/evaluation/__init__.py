"""Evaluators of the video-to-MuJoCo benchmark (PhysInOne + Physion v1.5). Driver: benchmark/evaluate.py.

v2          visible-motion evaluator v2: input-view 3D / 2D ADE and IoU of the moving objects (renders the
            prediction with `python -m evaluation.render` in a subprocess)
physics     physical quantities (physics_v6) measured identically on the GT and the prediction's passive rollout
secondview  2D ADE and IoU of the moving objects seen from a camera the agent never saw
gt          GT annotations (masks, poses); render: trusted passive rollout of the prediction; common: helpers
"""
