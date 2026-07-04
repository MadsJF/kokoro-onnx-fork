{
  buildPythonPackage,
  replaceVars,
  lib,
  stdenv,
  fetchPypi,
  hatchling,
  numpy,
  onnxruntime,
  espeak-ng,
  phonemizer,
}:
buildPythonPackage rec {
  pname = "kokoro_onnx";
  version = "0.5.0";

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-W+sV8IXigo7Y1JP3ksB5r4VxA6stzqoeESsXYFh6yWo=";
  };

  pyproject = true;
  build-system = [ hatchling ];

  patches = [
    # this replaces the need for espeakng-loader
    (replaceVars ./set-espeak-paths.patch {
      espeak-library-path = "${lib.getLib espeak-ng}/lib/libespeak-ng${stdenv.hostPlatform.extensions.sharedLibrary}";
      espeak-data-path = "${lib.getLib espeak-ng}/share/espeak-ng-data";
    })
  ];

  # remove espeakng-loader as dependency and phonemizer is already patched in nixpkgs
  postPatch = ''
    sed -i '/"espeakng-loader.*",/d' pyproject.toml
    substituteInPlace pyproject.toml --replace-fail "phonemizer-fork>=3.3.2" "phonemizer"
  '';

  doCheck = false;

  dependencies = [
    numpy
    onnxruntime
    espeak-ng
    phonemizer
  ];
}