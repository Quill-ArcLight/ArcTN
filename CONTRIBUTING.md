# Contributing

Issues and pull requests are welcome. The project maintainer is **liuxl**.
For a vulnerability, use the private contact in [SECURITY.md](SECURITY.md).

## License and contribution rights

This project is **source available, noncommercial and subject to public-source
requirements**, under [Arclight Noncommercial Source-Available License 1.0](LICENSE).
It is not OSI open source. Contributions to company-licensed files are submitted
under that same license. Contributions to separately licensed third-party files
must preserve and comply with those files' licenses and notices. Identify copied
or adapted material, its source, version and license in your pull request.

You retain your copyright. No copyright assignment or CLA is required. Submission
does not give the company a blanket right to relicense your contribution under
commercial terms. Future relicensing may require your additional permission.

## Signed contribution origin statement

Sign each contribution commit with `git commit -s`. Use a name you are entitled
to use and a working email address. The name and email become public Git data.
The sign-off must match the commit author's name and email:

```text
Signed-off-by: Your Name <your-email@example.org>
```

By adding this trailer and submitting the commit, you state that:

1. You created the contribution or have identified its external source and have
   the necessary permission to submit it under the applicable license described
   above, including any necessary employer permission.
2. You intend to grant the contribution under that license, and have disclosed
   any additional terms or rights that would prevent that grant.
3. You understand the contribution, attribution and sign-off will form a public
   record, and you have not knowingly included secrets or confidential material.

This is this project's contribution origin statement. It uses a sign-off
workflow similar to DCO, but **is not the standard DCO 1.1**: that text refers to
an open-source license, whereas this project restricts commercial use. A trailer
records a contributor's statement; it does not independently prove identity or
ownership. The maintainer still reviews provenance and licensing.

## Changes and validation

Describe the problem, change, test evidence and any effect on algorithm results.
Keep changes focused. Explain new dependency and data sources. For Rust changes:

```powershell
cargo fmt --all --check
cargo test --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings
```

For Python changes, build the extension and run `python -m pytest -q pybind/tests`.
Do not include private engine source or binaries, credentials or unrelated data.
Source-built wheels must not contain the private engine. Complete wheels need
separate engine redistribution approval and dependency review.

liuxl handles triage, repository administration, CI and releases. Changes whose
algorithmic correctness cannot be assessed may wait for suitable technical
review. An Issue or passing CI does not guarantee acceptance or a response time.
