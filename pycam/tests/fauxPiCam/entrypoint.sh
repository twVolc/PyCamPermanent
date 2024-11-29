#!/bin/sh
set -o errexit
set -o nounset

# We set ownership here in case the path is used as a volume
chown "${FTP_USER}":"${FTP_USER}" "/home/${FTP_USER}"

echo "${FTP_USER}:${FTP_PASS}" | /usr/sbin/chpasswd

exec /usr/sbin/proftpd --nodaemon