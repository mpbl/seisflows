import os as _os
import shutil as _shutil
import socket as _socket

from os.path import abspath, basename, isdir, isfile, join


def cat(src, *dst):
    f = open(src, 'r')
    contents = f.read()
    f.close()

    if not dst:
        print contents
    else:
        f = open(dst, 'w')
        f.write(contents)
        f.close()


def cd(path):
    _os.chdir(path)


def cp(src='', dst='', opt=''):
    if isinstance(src, (list, tuple)):
        if len(src) > 1:
            assert isdir(dst)

        for sub in src:
            cp(sub, dst)
        return

    if isdir(dst):
        dst = join(dst, basename(src))
        if isdir(dst):
            for sub in ls(src):
                cp(join(src, sub), dst)
            return

    if isfile(src):
        _shutil.copy(src, dst)

    elif isdir(src):
        _shutil.copytree(src, dst)


def hostname():
    return _socket.gethostname()


def ln(src, dst):
    if _os.path.isdir(dst):
        for name in _strlist(src):
            s = abspath(name)
            d = join(dst, basename(name))
            _os.symlink(s, d)
    else:
        _os.symlink(src, dst)


def ls(path):
    dirs = _os.listdir(path)
    for dir in dirs:
        if dir[0] == '.':
            dirs.remove(dir)
    return dirs


def mkdir(dirs):
    for dir in _strlist(dirs):
        if not _os.path.isdir(dir):
            _os.makedirs(dir)


def mkdir_gpfs(dirs):
    try:
        for dir in _strlist(dirs):
            if not _os.path.isdir(dir):
                _os.makedirs(dir)
    except:
        pass


def mv(src='', dst=''):
    if isinstance(src, (list, tuple)):
        if len(src) > 1:
            assert isdir(dst)
        for sub in src:
            mv(sub, dst)
        return

    if isdir(dst):
        dst = join(dst, basename(src))

    _shutil.move(src, dst)


def pwd():
    return _os.getcwd()


def rename(old, new, names):
    for name in names:
        if name.find(old) >= 0:
            _os.rename(name, name.replace(old, new))


def rm(path=''):
    for name in _strlist(path):
        if _os.path.isfile(name):
            _os.remove(name)
        elif _os.path.islink(name):
            _os.remove(name)
        elif _os.path.isdir(name):
            _shutil.rmtree(name)


def select(items, prompt=''):
    while True:
        if prompt:
            print prompt
        for i, item in enumerate(items):
            print("%2d) %s" % (i + 1, item))
        try:
            reply = int(raw_input().strip())
            status = (1 <= reply <= len(items))
        except (ValueError, TypeError, OverflowError):
            status = 0
        if status:
            return items[reply - 1]


def whoami():
    import getpass

    return getpass.getuser()


def _strlist(object):
    if not isinstance(object, list):
        return [object]
    else:
        return object
