//===- Trace.h - XRay Trace Abstraction -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines the XRay Trace class representing records in an XRay trace file.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_XRAY_TRACE_H
#define LLVM_XRAY_TRACE_H

#include <cstdint>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/XRay/XRayRecord.h"

namespace llvm {
namespace xray {

/// A Trace object represents the records that have been loaded from XRay
/// log files generated by instrumented binaries. We encapsulate the logic of
/// reading the traces in factory functions that populate the Trace object
/// appropriately.
///
/// Trace objects provide an accessor to an XRayFileHeader which says more about
/// details of the file from which the XRay trace was loaded from.
///
/// Usage:
///
///   if (auto TraceOrErr = loadTraceFile("xray-log.something.xray")) {
///     auto& T = *TraceOrErr;
///     // T.getFileHeader() will provide information from the trace header.
///     for (const XRayRecord &R : T) {
///       // ... do something with R here.
///     }
///   } else {
///     // Handle the error here.
///   }
///
class Trace {
  XRayFileHeader FileHeader;
  std::vector<XRayRecord> Records;

  typedef std::vector<XRayRecord>::const_iterator citerator;

  friend Expected<Trace> loadTraceFile(StringRef, bool);

public:
  /// Provides access to the loaded XRay trace file header.
  const XRayFileHeader &getFileHeader() const { return FileHeader; }

  citerator begin() const { return Records.begin(); }
  citerator end() const { return Records.end(); }
  size_t size() const { return Records.size(); }
};

/// This function will attempt to load XRay trace records from the provided
/// |Filename|.
Expected<Trace> loadTraceFile(StringRef Filename, bool Sort = false);

} // namespace xray
} // namespace llvm

#endif // LLVM_XRAY_TRACE_H