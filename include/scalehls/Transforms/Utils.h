//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#ifndef SCALEHLS_TRANSFORMS_UTILS_H
#define SCALEHLS_TRANSFORMS_UTILS_H

#include "scalehls/Analysis/Utils.h"

namespace mlir {
namespace scalehls {

using TileList = SmallVector<unsigned, 8>;

/// Apply loop perfection. Try to sink all operations between loop statements
/// into the innermost loop of the input loop band.
bool applyAffineLoopPerfection(AffineLoopBand &band);

/// Optimize loop order. Loops associated with memory access dependencies are
/// moved to an as outer as possible location of the input loop band. If
/// "reverse" is true, as inner as possible.
bool applyAffineLoopOrderOpt(AffineLoopBand &band, bool reverse = false);

/// Try to rectangularize the input band.
bool applyRemoveVariableBound(AffineLoopBand &band);

/// Apply optimization strategy to a loop band. The ancestor function is also
/// passed in because the post-tiling optimizations have to take function as
/// target, e.g. canonicalizer and array partition.
bool applyOptStrategy(AffineLoopBand &band, FuncOp func, TileList tileList,
                      unsigned targetII);

/// Apply optimization strategy to a function.
bool applyOptStrategy(FuncOp func, ArrayRef<TileList> tileLists,
                      ArrayRef<unsigned> targetIIs);

/// Apply loop tiling to the input loop band and sink all intra-tile loops to
/// the innermost loop with the original loop order. Return the location of the
/// innermost tile-space loop.
Optional<unsigned> applyLoopTiling(AffineLoopBand &band, TileList tileList);

/// Apply loop pipelining to the pipelineLoc of the input loop band, all inner
/// loops are automatically fully unrolled.
bool applyLoopPipelining(AffineLoopBand &band, unsigned pipelineLoc,
                         unsigned targetII);

/// Fully unroll all loops insides of a loop block.
bool applyFullyLoopUnrolling(Block &block);

} // namespace scalehls
} // namespace mlir

#endif // SCALEHLS_TRANSFORMS_UTILS_H