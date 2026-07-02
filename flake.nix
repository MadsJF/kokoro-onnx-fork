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
  let
    # 1. Define the overlay at the top level (system-agnostic)
    # This injects kokoro_onnx straight into Nixpkgs' Python infrastructure
    kokoro_overlay = final: prev: {
      pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
        (python-final: python-prev: {
          kokoro_onnx = python-final.callPackage ./nix/custom_pkgs/kokoro-onnx.nix {
            espeak-ng = final.espeak-ng;
          };
          fugashi = python-prev.fugashi.overridePythonAttrs (oldAttrs: {
            patches = (oldAttrs.patches or []) ++ [ ./nix/patches/fugashi_mecab_path.patch];
          });
        })
      ];
    };
  in
    {
      overlays.default = kokoro_overlay;
    }
    //
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ kokoro_overlay ];
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