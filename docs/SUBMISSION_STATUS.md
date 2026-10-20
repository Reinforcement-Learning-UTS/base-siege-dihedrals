# Migration and verification status

Prepared 20 October 2026 against corrected organisation main
`3299e10e4dfcfd5f80a6f3e52f969cdd3b2e772a`.

The first stage supplied the shared foundation and TRPO. Its complete 186-file
package matched the two original contribution commits. A separate layout fix
moved the project to the repository root and replaced the duplicated README.
Both original commits remain in history. The corrected root passed 218 tests,
with one expected research-stage skip, and TRPO CLI help.

This stage adds the remaining PPO/DDQN and V2/V3 research workflows, model catalogs,
replay utilities, notebooks, handbooks and preserved research evidence. It is a
prospective review and migration handoff, not a claim of new experiment execution
or sole authorship of the original code.

## Models and evidence

All 2,832 reviewed LFS objects, 10,613,541,750 bytes, were uploaded to the
organisation store. All object identities and sizes were checked. The 51 unique
final models were freshly downloaded and matched their original SHA-256 hashes.
Research ZIPs retain the exact original pointers, keeping binary payloads out of
manual handoffs. All ordinary evidence keeps its original bytes.

Use the final-model include list for targeted hydration. The full historical
archive is much larger than the 212.25 MB of unique final weights and should not
be downloaded automatically for an application build.

## Validation scope

The handoff is checked by applying it to an independent clone of its exact base,
verifying every file against the inventory, collecting and running the included
research tests, and importing the CLI entrypoints. A separate local check hydrates
only approved final weights and exercises all 51 CPU model/example paths.
The private package-validation receipt records exact results.

The earlier complete local preparation passed 459 regression tests and forty
independent DDQN runtime-parity cases. That is evidence for the complete local
preparation, not a claim that the application is already in this research stage.

Frozen scientific evaluators and DDQN training/recovery retain their strict
recorded Python 3.12.14 runtime. Extended research tests require that interpreter
and the selectively hydrated checkpoint weights. The CPU
inference adapter supports final Python 3.12 patches with the recorded package
versions and the matching Torch CPU wheel, with original tensor and file checks.
No historical runtime contract was falsified.

## Remaining work

The application stage adds the full Streamlit entrypoints and editor, complete
application regression checks, final documentation and CI integration. Hosting
then requires selective model hydration and verification on the actual Linux CPU
runtime. No hosted deployment or new final scientific evaluation is claimed.
