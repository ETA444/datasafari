#!/bin/bash

# ensure we're in the repository root
cd "$(git rev-parse --show-toplevel)" || exit

# update main branch first
git checkout main && git pull origin main

# fetch all branches from the remote repository and exclude 'origin/main'
branches=$(git branch -r | grep -v 'main' | sed 's/origin\///')

for branch in $branches; do
    # trim leading/trailing whitespace
    branch=$(echo "$branch" | tr -d '[:space:]')

    # skip if branch is main (safety check)
    if [[ "$branch" == "main" ]]; then
        continue
    fi

    # Extract the branch name without 'origin/' prefix for local operations
    local_branch=$(echo "$branch" | sed 's/origin\///')

    # check if local branch exists; if not, create it
    if ! git rev-parse --verify "$local_branch" > /dev/null 2>&1; then
        git branch --track "$local_branch" "origin/$local_branch"
    fi

    # checkout the branch and merge or rebase main branch
    git checkout "$local_branch" && \
    git merge main || {
        echo "Error: Merge conflict in $local_branch. Aborting."
        exit 1
    }

    # push changes to origin
    git push origin "$local_branch"
done

# Optionally, switch back to main at the end
git checkout main
