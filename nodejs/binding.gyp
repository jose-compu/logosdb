{
  "variables": {
    "openssl_fips%": ""
  },
  "targets": [
    {
      "target_name": "logosdb",
      "sources": [
        "src/node_logosdb.cpp",
        "deps/core/src/logosdb.cpp",
        "deps/core/src/storage.cpp",
        "deps/core/src/metadata.cpp",
        "deps/core/src/hnsw_index.cpp",
        "deps/core/src/wal.cpp",
        "deps/core/src/platform.cpp"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "deps/core/include",
        "deps/core/src",
        "deps/core/third_party"
      ],
      "cflags!": ["-fno-exceptions"],
      "cflags_cc!": ["-fno-exceptions"],
      "cflags_cc": [
        "-std=c++17",
        "-fexceptions"
      ],
      "conditions": [
        ["OS=='mac'", {
          "xcode_settings": {
            "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
            "CLANG_CXX_LIBRARY": "libc++",
            "MACOSX_DEPLOYMENT_TARGET": "11.0",
            "OTHER_CPLUSPLUSFLAGS": [
              "-std=c++17",
              "-fexceptions"
            ]
          }
        }],
        ["OS=='linux'", {
          "cflags_cc": [
            "-std=c++17",
            "-fexceptions"
          ],
          "ldflags": [
            "-Wl,--gc-sections"
          ]
        }],
        ["OS=='win'", {
          "msvs_settings": {
            "VCCLCompilerTool": {
              "ExceptionHandling": 1,
              "AdditionalOptions": ["/std:c++17"]
            }
          }
        }]
      ],
      "defines": [
        "NAPI_CPP_EXCEPTIONS",
        "NAPI_VERSION=8"
      ]
    }
  ]
}
