# Third-party and prior-license notices

Company-authorized ArcTN source in this revision uses the Arclight
Noncommercial Source-Available License 1.0 in LICENSE. This is not an
OSI-approved open-source license. Dependencies retain their own licenses;
the company license does not impose noncommercial restrictions on their files.

Earlier private development revisions contained developer-generated MIT and
Apache-2.0 license templates. The company has confirmed that these revisions
were never distributed externally. The old texts and attribution are retained
under THIRD_PARTY_LICENSES/ArcTN-prior-terms as development history, not as an
alternative license offer for this release. Separately identified third-party
material continues to retain its original license and permissions.

The Light/Heavy engine is maintained separately. Source-only builds do not
include it. This source license grants no permission to distribute or use
the proprietary engine. A complete wheel including it needs its own engine
terms and a dependency and binary-content review.

The Rust locks cover core and Python-extension dependencies separately,
including optional MPI build dependencies. A source dependency SBOM does not
identify the components inside an independently built engine, MPI installation,
or arbitrary external array backend. Their distributors must provide the
corresponding notices and source obligations for the actual artifacts.

## Retained dependency notices

`THIRD_PARTY_LICENSES/cargo-components.csv` lists the exact Cargo versions,
archive SHA-256 values, declared licenses and selected package-level licenses.
Both manifests and all platform/build/development dependencies are inventoried.
License texts and upstream copyright notices are kept in the matching directories.
Supplementary upstream texts are pinned to their Cargo-recorded Git commits in
`THIRD_PARTY_LICENSES/license-supplements.json`.

Some dependency archives have separately licensed examples or build tools:

- Crossbeam Channel's `examples/matching.rs` adapts `matching.go` by Stefan
  Nilsson under CC-BY-3.0. Its test notices also identify Rust and Go project
  material under MIT/Apache-2.0 and BSD-3-Clause. Retain LICENSE-THIRD-PARTY
  and the original authors' attribution when redistributing those files.
- `libffi-sys` bundles libffi under its permissive license. The separately
  identified build/test helpers in libffi/LICENSE-BUILDTOOLS use GPL-2.0;
  that notice expressly distinguishes those helpers from libffi itself.
  This source snapshot retains the notice text but does not vendor those
  helper programs. Redistributing a full dependency source archive requires
  retaining its helper sources and original licenses as well.

Package-level license choices do not replace these file-specific notices.
No third-party implementation or upstream license text was modified by this
release preparation; copied notice files retain their original bytes.
