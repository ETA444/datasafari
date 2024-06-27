
#!/bin/bash

# define the repository and backup directory
REPO_DIR="/c/dev/datasafari"
BACKUP_ROOT_DIR="/c/dev/repo-backups/datasafari"
BACKUP_DIR="$BACKUP_ROOT_DIR/$(date +%Y-%m-%d_%H-%M-%S)"

# create the backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# create a backup
cp -a "$REPO_DIR" "$BACKUP_DIR" && echo "Backup of repository created at $BACKUP_DIR" || echo "Backup failed"

# Backup retention policy: Keep only the latest 50 backups
# get the number of backup directories
backup_count=$(ls -d "$BACKUP_ROOT_DIR"/*/ | wc -l)
max_backups=50

# IF: more than the maximum number of backups, delete the oldest
if [ "$backup_count" -gt "$max_backups" ]; then
    # delete the oldest backups exceeding the max_backups limit
    ls -d "$BACKUP_ROOT_DIR"/*/ | head -n -$max_backups | xargs rm -rf
    echo "Old backups deleted, keeping only the latest $max_backups backups."
fi
