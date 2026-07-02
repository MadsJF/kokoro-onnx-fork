{
  description = "Flake for packaging kokoro-onnx-fork";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      python_kokoro_overlay = final: prev: {
        # Patch fugashi to add enviorment variables for UNIDIC_DICDIR and MECABRC_FILE 
        # (way of fixing some max path length for mecab)
        pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
          (python-final: python-prev: {
            fugashi = python-prev.fugashi.overridePythonAttrs (oldAttrs: {
              patches = (oldAttrs.patches or []) ++ [ ./nix/patches/fugashi_mecab_path.patch];
            });
          })
        ];
      };

      pkgs = import nixpkgs {
        inherit system;
        overlays = [ python_kokoro_overlay ];
      };
      pythonEnv = pkgs.python3;
    in {
      devShells.default = import ./nix/shell.nix {
        inherit pkgs;
      };
      packages.default = pythonEnv.pkgs.callPackage ./nix/custom_pkgs/kokoro-onnx.nix {
        # espeak-ng isn't a python package, so pass it explicitly from top-level pkgs
        espeak-ng = pkgs.espeak-ng;
      };
    });
}