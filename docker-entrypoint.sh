#!/bin/sh
# Whisper Transcription Manager container start (Dockerfile ENTRYPOINT).
#
# Started as root: writes /config/config.yaml from config.docker.yaml on the
# first start, hands the folders the app writes to the user it runs as, then
# runs the server as that user. The user is PUID:PGID, else the owner of
# /config — a folder of the host: what the app saves there stays the host
# user's (with rootless Docker or Podman that owner is root in the
# container, which is the host user, so it runs as root there).
set -e

if [ "$(id -u)" != 0 ]; then
    # Started as another user (docker run --user): nothing to hand over
    # 0600: a password typed into it later is nobody else's to read
    [ -e /config/config.yaml ] || (umask 077 && cp /app/config.docker.yaml /config/config.yaml)
    exec "$@"
fi

uid="${PUID:-$(stat -c %u /config)}"
gid="${PGID:-$(stat -c %g /config)}"

if [ ! -e /config/config.yaml ]; then
    (umask 077 && cp /app/config.docker.yaml /config/config.yaml)
    chown "$uid:$gid" /config/config.yaml
    echo "First start: /config/config.yaml written from config.docker.yaml"
fi

# Never /media: the library is the user's, mounted read-only. A folder
# that refuses the change (a network share) is left as it is.
for dir in /config /data /cache "$HOME"; do
    mkdir -p "$dir"
    if [ "$(stat -c %u:%g "$dir")" != "$uid:$gid" ]; then
        chown -R "$uid:$gid" "$dir" \
            || echo "Warning: could not hand $dir to $uid:$gid — it must be writable for it"
    fi
done

if [ "$uid" = 0 ]; then
    exec "$@"
fi
exec setpriv --reuid="$uid" --regid="$gid" --clear-groups -- "$@"
