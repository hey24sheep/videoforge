import sys

# light commands start without the render stack (cupy, opencv): instant, and no CUDA import warnings
_a = sys.argv[1:]
if not _a:
    from .ui import overview
    overview()
elif _a[0] in ('status', 'config'):
    from importlib import import_module
    sys.exit(import_module('.' + _a[0], 'vf').main(_a[1:]))
elif _a[0] == '--version':
    from .ui import version
    print(f'videoforge {version()}')
else:
    from .build import main
    main()
