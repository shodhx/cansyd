# SEU / Gearbox cross-domain validation - handoff

Goal: run the full CNSD pipeline on SEU gears (a non-bearing domain) to
demonstrate the provider interface is genuinely domain-agnostic. This is the
universality result.

## What's included (built + tested)

- `cnsd/physics/gear.py` - gear-mesh physics: GMF (= teeth x shaft-rate),
  shaft-rate sidebands, envelope-spectrum prominence. GMF math verified
  (20 teeth @ 1800 rpm -> 600 Hz).
- `cnsd/physics/providers/gear.py` - GearProvider, implements the same
  PhysicsProvider interface as BearingProvider. Maps the SEU 5-class taxonomy
  (Health/Chipped/Miss/Root/Surface) onto two evidence channels: localized tooth
  faults -> GMF sidebands; surface wear -> GMF harmonics.
- `cnsd/physics/providers/__init__.py` - registers domain 'gear'.
- `cnsd/builder.py` - config path: `domain.type: gear` builds the GearProvider
  from `physics.parameters.n_teeth_input` (+ optional n_teeth_output).
- `cnsd/physics/configs.py` - PhysicsConfig.bearing is now Optional (gear configs
  have no bearing geometry).
- `validate_seu.py` - the run script. Only `load_seu()` is dataset-specific.
- 3 new tests (gear math, interface conformance, registry) - 10/10 pass.

## Your job

1. Wire `load_seu()` in validate_seu.py - return X (n,1024), y (n,), cond (n,).
   SEU gearset: tab-separated, 8 channels, header ends at the 'Data' line; pick
   ONE channel and commit to it before seeing results (no channel cherry-picking).
2. Set `N_TEETH_INPUT` to the real SEU driving-gear tooth count (I used 20 as a
   placeholder - CONFIRM from the rig spec; the GMF is wrong if this is wrong).
3. Confirm `SEU_FS` (sampling rate) and the condition->rpm map.
4. Run `python validate_seu.py`, send me the output.

## Honest caveats (expect these, they are not bugs)

- The prominence threshold (3.0, same as bearings) will likely need tuning on
  real SEU signals - same story as the CWRU sweep. Expect a high INCONCLUSIVE
  rate at first; sweep tau on a held-out calibration split (NOT the test set, NOT
  the training set where the CNN memorizes - the lesson from #11).
- The Chipped/Miss/Root families all share the sideband channel, so the physics
  cannot distinguish *between* them (they look alike in the envelope spectrum -
  this is physically true, localized tooth faults all produce shaft-rate
  sidebands). The provider confirms "localized tooth fault" vs "surface wear" vs
  "health"; finer separation is the CNN's job. Report this honestly - it's a real
  property of gear physics, not a limitation to hide.
- N_TEETH_INPUT is the single most important parameter. If the GMF is off, every
  verdict is meaningless. Triple-check it.

## How it plugs in (no engine change)

The engine never sees gear-specific code - it calls the PhysicsProvider
interface. Same five layers, same diagnose() path, only the provider differs.
That's the universality claim, demonstrated.
