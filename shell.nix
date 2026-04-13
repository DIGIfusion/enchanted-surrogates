{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python313;
in
pkgs.mkShell {
  packages = with pkgs; [
    stdenv

    python
    python.pkgs.pip
    python.pkgs.virtualenv
    python.pkgs.numpy
    python.pkgs.torch

    gcc
    gfortran
    openblas
    pkg-config
    cmake
    ninja
    patchelf
  ];

  OPENBLAS = pkgs.openblas;

  shellHook = ''
    VENV=.venv

    if [ ! -d "$VENV" ]; then
      echo "📦 Creating virtual environment in $VENV"
      ${python.interpreter} -m venv --system-site-packages $VENV
      source $VENV/bin/activate
      pip install --upgrade pip setuptools wheel
      pip install -e .
    else
      source $VENV/bin/activate
    fi

    echo "🐍 Python: $(python --version)"
    echo "📍 venv: $VENV"
  '';
}

