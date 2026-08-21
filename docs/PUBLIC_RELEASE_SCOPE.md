# Public Release Scope

The GitHub release is permitted only after the current-tree and published-history
manifests are complete, every decision is resolved, hygiene checks pass, and no
row requires history removal. A `remove_from_history` decision stops this
workflow and requires a separately approved destructive-action plan.

The decision authority role is `repository_owner_publisher`. It records
Leonardo's GitHub source-release decision and risk acceptance; it is not legal
advice, specialist clearance, or authority to grant third-party rights. GitHub
source publication did not itself authorize a Hugging Face weight or adapter
release. That later release has its own exact owner decision, package manifest,
validation contract, and public record under `release/huggingface/`.

## Included candidates

- Leonardo-owned software, tests, demo code, CI, and executable configuration;
- specifically identified Leonardo-owned documentation and aggregate evidence;
- source and attribution metadata whose terms permit redistribution;
- processed corpus material only when its exact current and historical blobs
  have affirmative rights and privacy dispositions.

## Excluded artifacts

- model weights, checkpoints, adapters, optimizer and scheduler state;
- raw poems or openings used in evaluation, generations, candidates,
  preferences, votes, annotations, mappings, tensors, and caches;
- raw or interim corpus acquisitions and local token streams;
- archives, staging bundles, runtime state, or logs containing machine paths;
- any artifact with missing, pending, conflicting, or unresolved review.

Twenty-eight output-bearing or prompt-level annotation files were removed from
the current release tree because they embed exact evaluation prompts, raw
model outputs, private mappings, or output-level annotations. Their historical
blobs remain under the owner-approved no-rewrite decision, carry no
repository-wide reuse grant for embedded content, and are reviewed separately
from the five exact historical machine-path exceptions.

GitHub `v1.0.0` may contain only GitHub's automatic source snapshots. No manual
Release assets are allowed. Making the repository private cannot retract prior
clones, caches, mirrors, or independently owned forks.

## Current publication status

The GitHub repository and source-snapshot-only `v1.0.0` release are public. The
fail-closed current-tree and reviewed-history manifests passed with every
decision resolved; GitHub Packages were enumerated with `read:packages` and the
repository had zero attached packages. The selected Stage 1, Stage 2, and Stage
3 weights plus DPO adapter were reviewed and published later through the
separate fail-closed [Hugging Face release](https://huggingface.co/LPM93/teaching-transformers-classical-italian-sonnets).
That release does not change the GitHub rule prohibiting model weights or manual
assets in the `v1.0.0` Release surface.
