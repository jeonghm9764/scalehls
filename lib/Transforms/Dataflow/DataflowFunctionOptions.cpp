//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "scalehls/Transforms/Passes.h"
#include "scalehls/Transforms/TosaOpHelper.h"
#include "scalehls/Transforms/Utils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

#define DEBUG_TYPE "scalehls"

using namespace mlir;
using namespace scalehls;
using namespace hls;

static bool applyDataflowFunctionOptions(ModuleOp module, unsigned numTargets,
                                     StringRef outputPath) {
  // auto builder = OpBuilder(module);

  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](DataflowNodeOp node) {
      node.dump();
      for (auto &op : node.body().front().getOperations()) {
        if (auto conv2DOp = dyn_cast<tosa::Conv2DOp>(op)) {
          conv2DOp.dump();
        }
      }
    });
  }

  return false;
}

namespace {
struct DataflowFunctionOptions
    : public DataflowFunctionOptionsBase<DataflowFunctionOptions> {
  void runOnOperation() override {
    auto module = getOperation();
    applyDataflowFunctionOptions(module, numTargets, outputPath);
  }
};
} // namespace

std::unique_ptr<Pass> scalehls::createDataflowFunctionOptionsPass() {
  return std::make_unique<DataflowFunctionOptions>();
}
