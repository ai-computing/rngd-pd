#!/usr/bin/env bash
# Resolve the pure-Python dependencies the copied site-packages expect.
#
# This host keeps Python packages in three places — the furiosa venv, ~/.local,
# and /usr/local/lib/python3.10/dist-packages — and the vendor stacks import
# across all three. The third is 9 GB of unrelated packages, so instead of
# copying it we let the image tell us what it actually misses: import the
# stack, read the ModuleNotFoundError, pip install that one name, repeat.
# What survives is the minimal closure, baked as one layer.
set -euo pipefail

IMG=$1                # image to fix, committed back to the same tag
PY=$2                 # interpreter inside the image
shift 2
PROBE="$*"            # import line that must succeed

c=$(buildah from "$IMG")
trap 'buildah rm "$c" >/dev/null 2>&1 || true' EXIT

buildah run "$c" -- sh -c \
  'command -v pip3 >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends python3-pip && rm -rf /var/lib/apt/lists/*)' >/dev/null

buildah run "$c" -- bash -c "
set -e
for i in \$(seq 1 25); do
  if out=\$($PY -c '$PROBE' 2>&1); then echo IMPORTS_OK; exit 0; fi
  mod=\$(echo \"\$out\" | sed -n \"s/.*No module named '\\([A-Za-z0-9_]*\\).*/\\1/p\" | head -1)
  if [ -z \"\$mod\" ]; then echo \"\$out\" | tail -8; echo IMPORTS_FAILED; exit 1; fi
  case \"\$mod\" in                    # import name != distribution name
    attr)    pkg=attrs ;;
    yaml)    pkg=PyYAML ;;
    PIL)     pkg=Pillow ;;
    sklearn) pkg=scikit-learn ;;
    cv2)     pkg=opencv-python-headless ;;
    google)  pkg=protobuf ;;
    *)       pkg=\"\$mod\" ;;
  esac
  echo \"  + installing \$mod (pip: \$pkg)\"
  pip3 uninstall -y --quiet \"\$mod\" >/dev/null 2>&1 || true
  pip3 install --no-cache-dir --quiet \"\$pkg\" >/dev/null 2>&1 || \
  pip3 install --no-cache-dir --quiet \"\${pkg//_/-}\" >/dev/null 2>&1 || \
    { echo \"cannot pip install \$pkg\"; exit 1; }
done
echo TOO_MANY_ROUNDS; exit 1
"

buildah commit --rm "$c" "$IMG"
trap - EXIT
echo "FIXED $IMG"
