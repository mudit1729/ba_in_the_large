from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import os
import sys
import subprocess
import platform
import shutil
try:
    import pybind11
    PYBIND11_AVAILABLE = True
except ImportError:
    PYBIND11_AVAILABLE = False


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        # Check if the required dependencies are available
        missing_deps = []
        
        # Check for CMake
        try:
            subprocess.check_call(['cmake', '--version'])
        except (OSError, subprocess.SubprocessError):
            missing_deps.append("CMake")
            
        # Check for pybind11
        if not PYBIND11_AVAILABLE:
            missing_deps.append("pybind11")
            
        # Check for Ceres Solver (if pkg-config is available)
        try:
            ceres_check = subprocess.run(['pkg-config', '--exists', 'ceres'], 
                                         stderr=subprocess.PIPE)
            if ceres_check.returncode != 0:
                missing_deps.append("Ceres Solver")
        except (OSError, subprocess.SubprocessError):
            # pkg-config not available, we'll check for Ceres during CMake
            pass
        
        if missing_deps:
            print("WARNING: Some dependencies are missing: {}".format(", ".join(missing_deps)))
            print("The package will be installed without C++ acceleration.")
            print("To enable C++ acceleration, please install the missing dependencies and reinstall.")
            return
        
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        # Get pybind11 include directory from Python package
        if PYBIND11_AVAILABLE:
            pybind11_include = pybind11.get_include()
        else:
            return
        
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DPYBIND11_INCLUDE_DIR=' + pybind11_include
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''), self.distribution.get_version())
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            
        # Change directory to the C++ source
        sourcedir = os.path.abspath(ext.sourcedir)
        
        print("Building C++ extension with Ceres Solver...")
        print("  CMake args:", cmake_args)
        print("  Build directory:", self.build_temp)
        print("  Source directory:", sourcedir)
        print("  Output directory:", extdir)
        
        try:
            # Configure
            subprocess.check_call(['cmake', sourcedir] + cmake_args,
                                cwd=self.build_temp, env=env)
            
            # Build
            subprocess.check_call(['cmake', '--build', '.'] + build_args,
                                cwd=self.build_temp)
            
            # Verify the built extension file exists
            expected_ext = os.path.join(extdir, 'ba_cpp.so')
            if os.path.exists(expected_ext):
                print(f"✓ Successfully built extension {ext.name}")
            else:
                raise RuntimeError(f"Extension file {expected_ext} not found after build")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Error building extension: {e}")
            print("  The package will be installed without C++ acceleration.")
            print("  To troubleshoot, try building manually:")
            print(f"    mkdir -p {self.build_temp}")
            print(f"    cd {self.build_temp}")
            print(f"    cmake {sourcedir} {' '.join(cmake_args)}")
            print(f"    cmake --build . {' '.join(build_args)}")
            
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            print("  The package will be installed without C++ acceleration.")
            
        print()


setup(
    name="ba_in_the_large",
    version="1.0.0",
    author="Mudit Jain",
    author_email="muditj@example.com",
    description="Bundle Adjustment in the Large - Python implementation with C++ acceleration",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mudit1729/ba_in_the_large",
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pybind11>=2.6.0",
        ],
    },
    ext_modules=[
        CMakeExtension("ba_in_the_large.ba_cpp", sourcedir="src/cpp"),
    ],
    cmdclass={
        "build_ext": CMakeBuild,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)