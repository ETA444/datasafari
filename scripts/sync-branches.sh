
#!/bin/bash

# fetch all branches from the remote repository
branches=$(git branch -r | awk -F'/' '{print $NF}')

# exclude 'main' branch from the list
branches=$(echo "$branches" | grep -v 'main')

# loop through each branch and merge with main
for branch in $branches; do
    # trim leading/trailing whitespace
    branch=$(echo "$branch" | tr -d '[:space:]')

    # checkout the branch
    git checkout "$branch"
    # merge or rebase main branch
    git merge main
    # push changes to origin
    git push origin "$branch"
done
