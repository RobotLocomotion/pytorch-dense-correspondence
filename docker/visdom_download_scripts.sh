#!/bin/bash

set -euxo pipefail

timeout --preserve-status 10 sudo python -m visdom.server || echo "success getting visdom fonts"

