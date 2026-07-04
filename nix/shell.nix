{
  pkgs ? import <nixpkgs> {},
}: let
  pythonEnv = pkgs.python313;

  kokoro-onnx = pythonEnv.pkgs.callPackage ./custom_pkgs/kokoro-onnx.nix {};
in
  pkgs.mkShell {
    name = "kokoro-onnx-flake-env";

    buildInputs = [
      kokoro-onnx
    ];

    shellHook = ''
      export PHONEMIZER_ESPEAK_LIBRARY="${pkgs.espeak-ng}/lib/libespeak-ng.so"
      export ESPEAK_DATA_PATH="${pkgs.espeak-ng}/share/espeak-ng-data"
      export LD_LIBRARY_PATH="${pkgs.libsndfile}/lib:${pkgs.onnxruntime}/lib:$LD_LIBRARY_PATH"

      echo "❄️ Pure ONNX Flake Environment Activated Successfully!"
    '';
  }
    # export MIOPEN_FIND_MODE="FAST"
    # export MIOPEN_DEBUG_DISABLE_FIND_DB="1"