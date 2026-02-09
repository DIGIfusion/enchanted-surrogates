# Releases

Releases are accessed in [GitHub](https://github.com/DIGIfusion/enchanted-surrogates).

Usage is just like described [here](https://digifusion.github.io/enchanted-surrogates/#how-to-install), just use the downloaded (and extracted) folder as destination for commands instead of cloning the repository from git.

## Documentation for creating release (for developers with write or push access)

Releases are accessed in [GitHub](https://github.com/DIGIfusion/enchanted-surrogates). Only users with write or push access to project can create release. Releases are generally created from `main` branch.

## Semantic versioning

Workflow uses GitHub actions to create Releases on GitHub. It runs automatically when you push a Git tag that matches the pattern `vX.Y.Z` where `X.Y.Z` is using [Semantic Versioning](https://semver.org/). It matches `major.minor.patch` type of semantic versioning:

1. Major `X` is increased when something backwards incompatible is created
2. Minor `Y` is increased when functionality is added in a backward compatible manner 
3. Patch `Z` is incremented when backward compatible bug fixes are made.

Increasing `X` resets `Y` and `Z`  to zero, and increasing `Y` resets `Z` to zero. Each increment is always one.

**Version should be incremented for each release**

## Triggering release

Currently workflow is triggered in two steps: first `git tag` and `git push` commands in development environment, then the release is created from [Releases subpage](https://github.com/DIGIfusion/enchanted-surrogates/releases/)






### Step 1: Local triggering

This requires user to create tag name `vX.Y.Z` according to semantic versioning described above. Workflow activates `.github/workflows/release.yaml`.


In `release.yaml` file linting and testing is run, then if those succeed package is built, and if build succeeds content of  `dist/*` folder is included in release. 

Triggering is done in the branch, that is to be released.
Use below commands to trigger first part of release workflow:
```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

### Step: 2 GitHub release
Check on [Actions](https://github.com/DIGIfusion/enchanted-surrogates/actions) page, that tests, build and artifact creation succeeded.

Go to [Releases](https://github.com/DIGIfusion/enchanted-surrogates/releases) page to create a new release:

1. Click Draft a new release. 
2. Select the tag `vX.Y.Z` which was used for local triggering
3. Click Generate release notes and use it as a base
4. Modify release notes if necessary and publish








