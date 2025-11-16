"""Multi-Agent RL Algorithms"""

from .maddpg import MADDPG
from .comm_maddpg import CommMADDPG

__all__ = [
    'MADDPG',
    'CommMADDPG'
]
