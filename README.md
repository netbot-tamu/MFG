MFG
===
####An Implementation of a Multi-Layer Feature Graph


Dependencies
------------
For a complete guide to this project's dependencies, see [DEPS.md](DEPS.md).  A summary table is provided below for convenient reference.

Dependency | Version | Debian/Ubuntu Package/Deps | Mac OS X Homebrew Package/Deps | Special Instructions
-----------|---------|-----------------------|---------------------------|---------------------
[Qt5](http://qt-project.org)                          | 5.4+    | qt5-default           | qt5       | *1
[OpenGL](http://freeglut.sourceforge.net)             | 2.8.1   | freeglut3-dev         | freeglut  | 
[OpenCV](http://opencv.org)                           | 2.4.10+ | -                     | opencv    | *2
[LSD](http://www.ipol.im/pub/art/2012/gjmr-lsd)       | 1.5     | -                     | -         | included
[LevMar](http://users.ics.forth.gr/~lourakis/levmar)  | 2.6     | -                     | -         | included
[Eigen3](http://eigen.tuxfamily.org)                  | 3       | libeigen3-dev         | eigen     | 
[G2O](https://openslam.org/g2o.html)                  | -       | libqt4-dev            | qt        | included *3
[QGLViewer](http://www.libqglviewer.com)              | 2.6.2   | -                     | -         | included



1. Qt5 via Homebrew on OS X
   * Qt5 is not installed into the prefix by default.  Instead, you must use
   ```bash
   $ brew install qt5
   $ brew link --force qt5
   # Where <version> is the installed version of qt5 (5.4.1 at the time of writing)
   $ ln -s /usr/local/Cellar/qt5/<version>/mkspecs /usr/local/mkspecs
   ```
2. OpenCV on Ubuntu-based systems
   * The nonfree module has been removed to avoid potential patent issues (https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=734373)
   * This means OpenCV should be installed manually from source on Ubuntu systems
3. G2O
   * Note that g2o is included as a git submodule and will quick-build automatically (skipping the examples and extras), as a part of the Build Instructions section below, if not already installed.


Ensure all installation paths are defined appropriately in the cmake_modules directory's Find[Library].cmake files, if they were installed manually.  If the library's path cannot be found, add a line to that library's Find[Library].cmake file for your specific installation path, but please do not commit it to the repository unless it is a generic path (is not under your home directory or in an obscure location).

Build Instructions
------------------
First, make sure the dependencies and submodules are up to date.  For submodules, use
```bash
$ git submodule init
$ git submodule update
```

This project uses CMake (http://www.cmake.org), a cross-platform build system.
```bash
$ cd [mfg root directory]
$ mkdir build
$ cd build
$ cmake ..
$ make
```


To run it
---------
1. Modify the configuration file: config/mfgSettings.ini accordingly. 
   For example, if using KITTI 00 dataset, just copy the content of "mfgSettings-kitti00.ini" to "mfgSettings.ini"
2. Change to the bin directory (generated after successful build) and run mfg-nogui or mfg-gui
3. To visualize the result, use the "src/matlab/plot_mfg_traj.m".

