MFG Dependencies
================

Below is a list of all dependencies required, and links to relevant information such as downloads.  There are also instructions for installing the required packages on a Ubuntu system, specifically, but which should also work on most Debian-based systems.
* Qt5: http://qt-project.org/
   * Note that using Homebrew on OS X, Qt5 is not installed into the prefix by default.  Instead, you must use
   ```bash
   # Debian-based Linux distributions:
   $ sudo apt-get install qt5-default
   # OS X using Homebrew requires some special steps to deal with a bug in Qt/Homebrew
   $ brew install qt5
   $ brew link --force qt5
   # Where <version> is the installed version of qt5 (5.4.1 at the time of writing)
   $ ln -s /usr/local/Cellar/qt5/<version>/mkspecs /usr/local/mkspecs
   ```
* OpenGL: freeglut (http://freeglut.sourceforge.net/)
   ```bash
   $ sudo apt-get install freeglut3-dev
   ```
* OpenCV: http://opencv.org
   * Note that on Ubuntu-based systems, the nonfree module has been removed to avoid potential patent issues (https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=734373)
   * This means OpenCV should be installed manually from source on Ubuntu systems
* Line Segment Detector from IPOL (Image Processing On Line)
   * Website link: http://www.ipol.im/pub/art/2012/gjmr-lsd/
   * Download link: http://www.ipol.im/pub/art/2012/gjmr-lsd/lsd_1.6.zip
   * A copy of the LSD project (version 1.5) resides in this repository and will build with MFG, and includes a custom CMakeLists.txt in order to compile with our code and link correctly
   * *Version 1.5 is required for the current state of this project*
      * The latest version (1.6), moved many of the header's declarations out of the lsd.h and into lsd.c
* LevMar, an implementation of Levenberg-Marquardt Nonlinear Least Squares Algorithms
   * Website link: http://users.ics.forth.gr/~lourakis/levmar/
   * Download link: http://users.ics.forth.gr/~lourakis/levmar/levmar-2.6.tgz
   * Requires lapack, f2c and OpenBlas
   ```bash
   $ sudo apt-get install liblapack-dev f2c libopenblas-base libopenblas-dev
   ```
   * Note that some issues have arisen during the LevMar compilation regarding an `undefined reference to symbol 'exp@@GLIBC...'`, which can be fixed by adding ` -lm` to the `TARGET_LINK_LIBRARIES` line in the levmar CMakeLists.txt file, making it look like:
   ```CMake
   TARGET_LINK_LIBRARIES(lmdemo ${LIBS} -lm)
   ```
   * Also note that levmar-2.6 has been included and will build with MFG, and contains a slightly modified CMakeLists.txt from the original (including the above change) in order to compile with our code and link correctly
* Eigen3: http://eigen.tuxfamily.org/
   ```bash
   $ sudo apt-get install libeigen3-dev
   ```
* g2o: A General Framework for Graph Optimization
   * Website link: https://openslam.org/g2o.html
   * GitHub download link: https://github.com/RainerKuemmerle/g2o
   * Note that the github code includes CSparse (http://people.sc.fsu.edu/~jburkardt/c_src/csparse/csparse.html)
   * On a UNIX system:
   ```bash
   $ git clone https://github.com/RainerKuemmerle/g2o
   $ cd g2o
   $ mkdir build
   $ cd build
   # for a full build, use the default configuration
   $ cmake ..
   # or for a quick build, skip the bundled apps and examples
   $ cmake .. -DG2O_BUILD_APPS=OFF -DG2O_BUILD_EXAMPLES=OFF
   # then make the project
   $ make
   $ make install
   ```
   * Note that g2o is included as a git submodule and will quick-build automatically, as a part of the Build Instructions section below if not already installed.
      * Thus the above steps can be skipped safely.
* QGLViewer: A simplified OpenGL visualization toolkit based on Qt
   * Website link: http://www.libqglviewer.com
   * GitHub download link: https://github.com/GillesDebunne/libQGLViewer
   * The current version (2.6.2) is included as a submodule

Ensure all installation paths are defined appropriately in the cmake_modules directory's Find[Library].cmake files, if they were installed manually.  If the library's path cannot be found, add a line to that library's Find[Library].cmake file for your specific installation path, but please do not commit it to the repository unless it is a generic path (is not under your home directory or in an obscure location).

