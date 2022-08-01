//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "scalehls/Transforms/Passes.h"
#include "scalehls/Transforms/Utils.h"

using namespace mlir;
using namespace scalehls;

namespace {
struct Conv2DOpRewritePattern : public OpRewritePattern<tosa::Conv2DOp> {
  using OpRewritePattern<tosa::Conv2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::Conv2DOp op,
                                PatternRewriter &rewriter) const override {
    if (op.input().getDefiningOp<tosa::TransposeOp>()) {
      if (auto func = dyn_cast<FuncOp>(op->getParentRegion()->getParentOp())) {
        auto input = func.front().addArgument(op.input().getType(), op.getLoc());
        op.inputMutable().assign(input);
        auto resultTypes = func.front().getTerminator()->getOperandTypes();
        auto inputTypes = func.front().getArgumentTypes();
        func.setType(rewriter.getFunctionType(inputTypes, resultTypes));
      }
    }

    if (op.weight().getDefiningOp()) {
      if (auto func = dyn_cast<FuncOp>(op->getParentRegion()->getParentOp())) {
        auto weight = func.front().addArgument(op.weight().getType(), op.getLoc());
        op.weightMutable().assign(weight);
        auto resultTypes = func.front().getTerminator()->getOperandTypes();
        auto inputTypes = func.front().getArgumentTypes();
        func.setType(rewriter.getFunctionType(inputTypes, resultTypes));
      }
    }

    if (op.bias().getDefiningOp()) {
      if (auto func = dyn_cast<FuncOp>(op->getParentRegion()->getParentOp())) {
        auto bias = func.front().addArgument(op.bias().getType(), op.getLoc());
        op.biasMutable().assign(bias);
        auto resultTypes = func.front().getTerminator()->getOperandTypes();
        auto inputTypes = func.front().getArgumentTypes();
        func.setType(rewriter.getFunctionType(inputTypes, resultTypes));
      }
    }

    return success();
  }
};
} // namespace

namespace {
struct MatMulOpRewritePattern : public OpRewritePattern<tosa::MatMulOp> {
  using OpRewritePattern<tosa::MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    if (op.b().getDefiningOp()) {
      if (auto func = dyn_cast<FuncOp>(op->getParentRegion()->getParentOp())) {
        auto weight = func.front().addArgument(op.b().getType(), op.getLoc());
        op.bMutable().assign(weight);
        auto resultTypes = func.front().getTerminator()->getOperandTypes();
        auto inputTypes = func.front().getArgumentTypes();
        func.setType(rewriter.getFunctionType(inputTypes, resultTypes));
      }
    }
    
    return success();
  }
};
} // namespace

namespace {
struct AddOpRewritePattern : public OpRewritePattern<tosa::AddOp> {
  using OpRewritePattern<tosa::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::AddOp op,
                                PatternRewriter &rewriter) const override {
    if (op.input1().getDefiningOp<tosa::ConstOp>()) {
      if (auto func = dyn_cast<FuncOp>(op->getParentRegion()->getParentOp())) {
        auto weight = func.front().addArgument(op.input1().getType(), op.getLoc());
        op.input1Mutable().assign(weight);
        auto resultTypes = func.front().getTerminator()->getOperandTypes();
        auto inputTypes = func.front().getArgumentTypes();
        func.setType(rewriter.getFunctionType(inputTypes, resultTypes));
      }
    }

    if (op.input2().getDefiningOp<tosa::ConstOp>()) {
      if (auto func = dyn_cast<FuncOp>(op->getParentRegion()->getParentOp())) {
        auto weight = func.front().addArgument(op.input2().getType(), op.getLoc());
        op.input2Mutable().assign(weight);
        auto resultTypes = func.front().getTerminator()->getOperandTypes();
        auto inputTypes = func.front().getArgumentTypes();
        func.setType(rewriter.getFunctionType(inputTypes, resultTypes));
      }
    }
    
    return success();
  }
};
} // namespace

namespace {
struct TosaConstToArgument
    : public TosaConstToArgumentBase<TosaConstToArgument> {
  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();
    auto builder = OpBuilder(module);

    mlir::RewritePatternSet patterns(context);
    patterns.add<Conv2DOpRewritePattern>(context);
    patterns.add<MatMulOpRewritePattern>(context);
    patterns.add<AddOpRewritePattern>(context);
    (void)applyPatternsAndFoldGreedily(module, std::move(patterns));

    // Order function arguments
    module.walk([&](FuncOp func) {
      auto numArguments = func.getNumArguments();
      func.walk([&](Operation *op) {
        for (auto operand : op->getOperands()) {
          if (!operand.getDefiningOp()) {
            auto newArgument =
                func.front().addArgument(operand.getType(), func.getLoc());
            operand.replaceAllUsesWith(newArgument);
          }
        }
      });
      for (unsigned idx = 0; idx < numArguments; idx++) {
        func.front().eraseArgument(0);
      }
      func.setType(builder.getFunctionType(
          func.front().getArgumentTypes(),
          func.back().getTerminator()->getOperandTypes()));
    });
  }
};
} // namespace

std::unique_ptr<Pass> scalehls::createTosaConstToArgumentPass() {
  return std::make_unique<TosaConstToArgument>();
}
