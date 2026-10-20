Gramicidin A in DMPC: equilibration + unbiased production
=========================================================

Needs only Python with OpenMM (>= 8, CUDA build) and numpy. No other files.

1. Copy this directory to the workstation and check it:
       PYTHON=python PLATFORM=CUDA ./launch_ga.sh --ns 0.01 --equil-scale 0.01
   (a ~minute end-to-end test; delete equilibrated.xml, ga_traj.npz,
   production_state.xml, equil.json afterwards)

2. Start the real run:
       PYTHON=python PLATFORM=CUDA ./launch_ga.sh --ns 100 --device 0
   It detaches; log out freely.  tail -f run.log  to watch.
   ~1 ns of staged equilibration runs first. If its checks fail
   (equil.json), production does NOT start -- send equil.json back.

3. Interrupted (reboot, killed)? Run the SAME command again. It resumes
   from the last checkpoint (every 200 ps) without duplicating frames.

4. Copy back:  ga_traj.npz  equil.json  run.log
   then locally:  gramicidin.load_run("ga_traj.npz")
