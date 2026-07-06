# OARM Reference Notes

## FASTER

Repository: https://github.com/mit-acl/faster

What OARM takes inspiration from:

- Free-known vs unknown distinction: unknown space is not treated as automatically forbidden; the risk comes from insufficient reaction margin.
- Time allocation idea: candidate execution time should express fast progress, cautious probing, braking, or yielding behavior.
- Contingency awareness: aggressive progress should be balanced against whether the vehicle can observe, replan, or stop in time.

What OARM does not borrow:

- No FASTER C++/ROS/Gurobi optimizer code is imported into this project.
- No YOPO baseline files are edited to host FASTER-style behavior.
- The historical `backup_logit` tensor is treated as a learned stopping/yield feasibility prior inside `OARM/`.
- OARM does not claim FASTER's certified backup trajectory guarantee.

Recommended paper boundary:

> FASTER solves safety through explicit map-based optimization and a guaranteed backup trajectory; OARM learns reaction-margin-aware primitive scoring from depth without online mapping.

Engineering rule:

- Keep `YOPO/` reproducible as the original baseline.
- Put all OARM-only modules under `OARM/`.
- If FASTER is later run as a traditional baseline, clone/build it outside `YOPO/` and record its commit and parameters separately.
