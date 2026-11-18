# Complete native source release delta

`gadget4-full-source/full-source.patch` is a complete delta from the retained
GADGET-4 commit `6fb393b5e25907f2ff06211f67d341c5e40e90d1` to the portable
native files shipped under `core/software/gadget4`. The baseline is retained
under `core/software/upstream/gadget4`, including its upstream license.
`core/software/upstream/gadget4.provenance.json` records the repository, commit
and all retained upstream file hashes. The baseline was obtained through Git
transport and a detached checkout of the pinned commit.

The full patch includes new source files, changes, deletions and executable-mode
changes. The manifest binds complete baseline and target file inventories, not
only a hand-picked subset of modified files. Apply its recorded strip count of
2; the patch headers contain `a/pristine/` and `b/shipped/` components. Historic
individual patches remain useful provenance, but must not also be applied on top
of this complete delta.

Portable packaging regenerates the full delta from the actual current native
sources into the archive. It checks that the baseline still matches its retained
upstream provenance and that the two native file inventories match the archive
inventory. This prevents an old patch manifest from standing in for later source
changes. Generated patch data are inserted into the ZIP without mutating the
working source checkout.

Clean Linux release profiles can use this recipe:

    {"bundle_manifest": "core/patches/gadget4-full-source/manifest.json"}

The explicit clean release runner applies the complete delta to a copied pristine
tree, compares all target files and rejects extra or missing portable native
files. Patch invocation receipts are kept outside the reconstructed source tree.
These controls are implemented for later execution; no patch application,
package validation, native builds or Linux release controls were run while
authoring this bundle.
