# History
The format used in the HISTORY file is as follows:
1. version number
2. branch name/purpose of changes
3. list of changes

## [TODO]
- Configure setup.py `chore/setup-config-files`

## `pre-release` 0.2.0
### feature/explorer/explore_df
- Create `explore_df.py` module
  - Code preliminary idea
  - Add modularity to if statements
  - Add ability to customize underlying pandas methods using **kwargs
  - Add utility function `filter_kwargs` that filters which kwarg is appropriate for which method to avoid TypeErrors
  - Write docstring and expand it
  - ...

### chore/repo-structure
- `requirements_dev.txt` renamed to `requirements.txt`

## `pre-release` 0.1.5
### chore/repo-utility-scripts
- Create `auto-backup_local.sh`
  - Bash script that creates a local backup of the repo
- Create `sync-branches`
  - Bash script that automates the process of syncing branches to the main branch (using merge)

### chore/setup-config-files
- Cut down cookie cutter template to appropriate essentials
- Configure all misc files:
  - Configure `travis.yml`
  - Configure `setup.cfg`
  - Configure text files
  - Configure `Makefile`
  - Configure `tox.ini`
