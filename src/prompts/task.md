You are responsible for the real2sim reconstruction of one and only one video. The input is in input/ of the current working directory; write the outputs to output/.

1. Deliver an editable, executable MuJoCo scene that recovers the key objects, the geometric layout, the initial state, the real supports and connections, and the main order of interactions and directions of motion. A complete reproduction of the appearance is not required.

2. The only input is input/video.mp4. Read the video's own frame timestamps, frame rate and duration, and use the whole video and its timeline. Observe the whole clip, more densely around interactions. Do not cut segments that are hard to reproduce, move the simulation start, or change the playback speed or time scale to pass the evaluation.

3. The final motion must come from an explicit initial state and continuous physics simulation. Forbidden: trajectory playback, dynamic pose keyframes, corrections during the run, scheduled impulses, time-switched external forces, active actuators, and switching collisions on or off by time. Do not replace real mechanisms with invisible supports or arbitrarily locked degrees of freedom. Real joints, fixed bases, passive physical connections, springs and damping need video evidence or a clearly labelled mechanism inference. Do not substitute initial penetration, abnormal initial velocities or fictitious stored energy for drives that appear later in the video.

4. Geometry, physical parameters and evidence-backed initial states may be adjusted offline; during the final run the model parameters are fixed and the simulation runs continuously from the initial state. Offline fitting may rerun from the initial state many times. Keep the fitting code and the final run code apart; the final run must not depend on per-frame target trajectories or optimizers. Saving the simulated trajectory for evaluation is allowed; the drive of the final run must come from physical mechanisms.

5. Reconstruct from the input video.

6. When a constraint cannot be met or a mechanism is missing, keep the reconstruction you have; do not ask for the constraints to be relaxed, and do not trade a forbidden drive for success.

7. Write files only in the current working directory. Do not modify the host's persistent configuration, other tasks, the global Python environment or shared data. Use the Python given by the environment variable REAL2SIM_PYTHON (it has MuJoCo, NumPy, SciPy, OpenCV, Pillow and Matplotlib) and FFmpeg.

8. output/ contains at least:
   - scene.xml and the assets it needs: model parameters fixed, every reference inside output/.
   - initial_state.json: {"qpos": [...], "qvel": [...], "time": 0.0}; lengths matching the model's degrees of freedom, set only once, at the start of the run.
   - parameters.json: the final physical parameters, their units and the evidence / assumptions.
   - run.py: supports --headless --duration SECONDS --output DIR; after loading the scene and the initial state it steps the physics continuously, without loading the reference video, target trajectories, optimizers or files of other tasks. It may offer a render option.
   - preview.mp4: the whole simulation rendered on the input timeline, for external evaluation.

9. tools/ in the working directory has two GPU foundation-model tools: Pi3X (tools/pi3x_recon) and SAM3 (tools/sam3_track). They run on a GPU outside the sandbox and can be executed directly. The reconstruction must use both tools: call Pi3X and SAM3 and use their results in the reconstruction. The working directory also has a set of optional helper commands `"$REAL2SIM_PYTHON" -m r2s ...` (video timeline and contact sheets, GPU calls, reading Pi3X / SAM3 results, 2D tracks, simulating on the video timeline and comparing with the tracks, the run.py template and preview.mp4, the deliverable format check); read skill/SKILL.md before you start (a reference of the commands and the Python interface; it does not prescribe a reconstruction workflow). Whether and how you use them is up to you. skill/, r2s/ and tools/ are read-only tools, not deliverables; do not modify them. These tools do not change any requirement of items 1-8.
