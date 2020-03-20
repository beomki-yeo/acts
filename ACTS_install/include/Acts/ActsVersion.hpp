#ifndef ActsVersion_h
#define ActsVersion_h

#include <string_view>

//  Caution: this is the only Acts header that is guaranteed
//  to change with every Acts release. Including this header
//  will cause a recompile every time a new Acts version is
//  used.

// clang-format off
namespace Acts {
constexpr unsigned int VersionMajor = 9u;
constexpr unsigned int VersionMinor = 9u;
constexpr unsigned int VersionPatch = 9u;
constexpr unsigned int Version
    = 10000u * VersionMajor + 100u * VersionMinor + VersionPatch;
constexpr std::string_view CommitHash = "26c5717684d46bf36c071ba2c89e5272f17cf814-dirty";
constexpr std::string_view CommitHashShort = "26c571768-dirty";
}  // namespace Acts
// clang-format on

// for backward compatibility
#define ACTS_VERSION Acts::Version

#endif  // ActsVersion_h
