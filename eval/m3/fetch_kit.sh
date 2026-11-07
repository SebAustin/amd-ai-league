#!/bin/sh
# Download the official Match 3 starter kit into eval/m3/kit (git-ignored) and recreate the
# empty directory and the chmod-000 file a zip cannot carry.
set -e
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$here/kit"
curl -sSL -o "$here/kit/kit.zip" https://storage.googleapis.com/lablab-static-eu/share/mc3-starter-kit.zip
unzip -q -o "$here/kit/kit.zip" -d "$here/kit"
sh "$here/kit/mc3-starter-kit/setup.sh"
