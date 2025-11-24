{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  # Use an environment variable to explicitly tell the dynamic linker where to look.
  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";

  buildInputs = [
    pkgs.python3
    pkgs.stdenv.cc.cc.lib # Explicitly pull in the C++ libraries
    pkgs.gcc               # Keep this for general compatibility
  ];

  shellHook = ''
    echo "Nix dependencies (including LD_LIBRARY_PATH) are now configured."
    echo "Activate your Python virtual environment:"
    echo "source my_project_env/bin/activate"
  '';
}

# nix-shell
# source my_project_env/bin/activate
# pip install scikit-learn streamlit yfinance matplotlib tensorflow
# streamlit run app_interactiv.py