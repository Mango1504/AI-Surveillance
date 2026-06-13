"""Setup script for metropolis_cpp C++ extensions using CMake build."""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """A CMake-based extension module."""

    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    """Custom build_ext command that drives CMake."""

    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE=Release",
        ]

        build_args = ["--config", "Release"]

        # Set parallel build jobs
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_args += [f"--parallel", f"{os.cpu_count()}"]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args],
            cwd=build_temp,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args],
            cwd=build_temp,
            check=True,
        )


setup(
    name="metropolis_cpp",
    version="0.1.0",
    author="AI Surveillance Team",
    description="C++ CUDA extensions for NVIDIA Metropolis integration",
    long_description="High-performance CUDA preprocessing, NMS, and risk engine extensions with pybind11 bindings.",
    ext_modules=[CMakeExtension("metropolis_cpp")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.8",
)
