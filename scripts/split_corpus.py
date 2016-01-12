import argparse
import os
from posix import mkdir
from shutil import copyfile

desc = """Split a directory of files into "train", "dev" and "test" directories.
All files not in either "train" or "dev" will go into "test".
"""
TRAIN_DEFAULT = 290
DEV_DEFAULT = 35
# TEST on all the rest


def copy(src, dest, link=False):
    if link:
        try:
            os.symlink(src, dest)
        except (NotImplementedError, OSError):
            copyfile(src, dest)
    else:
        copyfile(src, dest)


def split_passages(directory, train=TRAIN_DEFAULT, dev=DEV_DEFAULT, link=False):
    filenames = sorted(os.listdir(directory))
    directory = os.path.abspath(directory)
    if not directory.endswith(os.sep):
        directory += os.sep
    for subdirectory in "train", "dev", "test":
        if not os.path.exists(directory + subdirectory):
            mkdir(directory + subdirectory)
    print("%d files to split" % len(filenames))
    print_format = "Creating link in %s to: " if link else "Copying to %s: "
    print(print_format % "train", end="", flush=True)
    for f in filenames[:train]:
        copy(directory + f, directory + "train" + os.sep + f, link)
        print(f, end=" ", flush=True)
    print()
    print(print_format % "dev", end="", flush=True)
    for f in filenames[train:train + dev]:
        copy(directory + f, directory + "dev" + os.sep + f, link)
        print(f, end=" ", flush=True)
    print()
    print(print_format % "test", end="", flush=True)
    for f in filenames[train + dev:]:
        copy(directory + f, directory + "test" + os.sep + f, link)
        print(f, end=" ", flush=True)
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("directory", default=".", nargs="?",
                        help="directory to split (default: current directory)")
    parser.add_argument("-t", "--train", default=TRAIN_DEFAULT,
                        help="size of train split (default: %d)" % TRAIN_DEFAULT)
    parser.add_argument("-d", "--dev", default=DEV_DEFAULT,
                        help="size of dev split (default: %d)" % DEV_DEFAULT)
    parser.add_argument("-l", "--link", action="store_true",
                        help="create symbolic link instead of copying")
    args = parser.parse_args()

    split_passages(args.directory, args.train, args.dev, link=args.link)
