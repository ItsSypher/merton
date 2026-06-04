# conda-forge feedstock recipe

This directory holds the `meta.yaml` we'll submit to
[conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes).
Pin it to the latest released version on PyPI (currently `1.0.1`).

## Submitting

1. Fork `conda-forge/staged-recipes`.
2. Create a branch `add-merton`.
3. Copy `merton/recipe/meta.yaml` into `recipes/merton/meta.yaml`.
4. Update the `sha256:` placeholder by running:
   ```bash
   pip download merton==1.0.1 --no-deps --no-binary :all: -d /tmp/m && \
   shasum -a 256 /tmp/m/merton-1.0.1.tar.gz
   ```
5. Open the PR. Reviewers will check the recipe; once merged, a feedstock
   repository at `conda-forge/merton-feedstock` is auto-created.

## Maintenance

After the feedstock is live, version bumps land via `regro/cf-scripts`
automation — usually a PR within an hour of a new PyPI release. Manual
intervention is only needed when dependency requirements change.

Recipe maintainers are listed in `meta.yaml::extra.recipe-maintainers`.
