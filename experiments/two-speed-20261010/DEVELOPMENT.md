# Development checks before registration

All data generated during development used the separate 12900000 seed range.
The initial seed-isolation test caught inherited `worlds` metadata and an
inherited bootstrap seed in the development fixture; those were removed or
replaced. The added end-to-end runner test then caught a missing `test_seeds`
field in its reduced one-world fixture. Both failures were corrected before
registered generation. The failed local logs are retained, with digests in
provenance/two-speed-development.json; their host-specific traceback paths are
not published.

The final pre-registration suite passed 238 tests: all 206 existing release
tests and 32 new tests, including every-policy leakage invariance and history
reload, bounded/disjoint search, world seed isolation, validation isolation,
deterministic recurrent fitting, independent-world bootstrap, and an entire
small development run plus refit/reselection/replay. Registered six-world
results require a separate full replay; passing development tests is not a
scientific qualification.
