"""videoforge: generic, cached, GPU video pipeline for YouTube (16:9) and Shorts/Reels (9:16) episodes."""
from . import config as _config
_config.apply()   # vf.config.json -> os.environ before any module reads its VF_* setting
