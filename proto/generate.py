#!/usr/bin/env python
"""Generate Python protobuf bindings from analytics.proto.

Usage:
    pip install grpcio-tools
    python proto/generate.py

This script uses grpc_tools.protoc to compile the .proto file into Python
bindings without requiring a system-level protoc installation.
"""
import os
import sys


def main():
    """Generate Python protobuf bindings."""
    try:
        from grpc_tools import protoc
    except ImportError:
        print(
            "Error: grpcio-tools is not installed.\n"
            "Install it with: pip install grpcio-tools\n"
            "Or: conda install grpcio-tools"
        )
        sys.exit(1)

    proto_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(
        os.path.dirname(proto_dir),
        "surveillance-app",
        "backend",
        "metropolis",
        "proto_generated",
    )
    os.makedirs(output_dir, exist_ok=True)

    # Generate Python protobuf bindings (messages + type stubs)
    result = protoc.main([
        "grpc_tools.protoc",
        f"--proto_path={proto_dir}",
        f"--python_out={output_dir}",
        f"--pyi_out={output_dir}",
        "analytics.proto",
    ])

    if result != 0:
        print(f"Error: protoc exited with code {result}")
        sys.exit(result)

    # Create __init__.py if it doesn't exist
    init_path = os.path.join(output_dir, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write('"""Generated protobuf bindings for analytics schema."""\n')

    print(f"Successfully generated Python protobuf bindings in: {output_dir}")
    print("Generated files:")
    for fname in sorted(os.listdir(output_dir)):
        if fname.endswith((".py", ".pyi")):
            print(f"  {fname}")


if __name__ == "__main__":
    main()
