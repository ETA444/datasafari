#!/bin/bash

# ensure we're in the repository root
cd "$(git rev-parse --show-toplevel)" || exit

# update main branch first
git checkout main && git pull origin main

# fetch all branches from the remote repository
branches=$(git branch -r | grep -v 'main' | awk -F'/' '{print $NF}')

for branch in $branches; do
    # trim leading/trailing whitespace
    branch=$(echo "$branch" | tr -d '[:space:]')

    # skip if branch is main (safety check)
    if [ "$branch" == "main" ]; then
        continue
    fi

    # check if local branch exists; if not, create it
    if ! git rev-parse --verify "$branch" > /dev/null 2>&1; then
        git branch --track "$branch" "origin/$branch"
    fi

    # checkout the branch and merge or rebase main branch
    git checkout "$branch" && \
    git merge main || {
        echo "Error: Merge conflict in $branch. Aborting."
        exit 1
    }

    # push changes to origin
    git push origin "$branch"
done

# checkout main branch at the end
git checkout main
