import logging
from logging import INFO

import os

"""
Setting up the basic logger for the project
"""

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", INFO),
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)

log = logging.getLogger("evaluator")
