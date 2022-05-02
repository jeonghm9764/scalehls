//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "scalehls/Transforms/Passes.h"
#include "scalehls/Transforms/Utils.h"

using namespace mlir;
using namespace scalehls;
using namespace hls;

struct ConvHelper {
  int64_t inCh;
  int64_t inSize;
  int64_t outCh;
  int64_t outSize;
  int64_t kernelSize;
  int64_t pad;
  int64_t stride;
  int64_t dilation;

  ConvHelper() {
    inCh = 0;
    inSize = 0;
    outCh = 0;
    outSize = 0;
    kernelSize = 0;
    pad = 0;
    stride = 0;
    dilation = 0;
  }

  ConvHelper(int64_t _inCh, int64_t _inSize, int64_t _outCh, int64_t _outSize,
             int64_t _kernelSize, int64_t _pad, int64_t _stride,
             int64_t _dilation) {
    inCh = _inCh;
    inSize = _inSize;
    outCh = _outCh;
    outSize = _outSize;
    kernelSize = _kernelSize;
    pad = _pad;
    stride = _stride;
    dilation = _dilation;
  }

  ConvHelper(tosa::Conv2DOp op) {
    auto inType = op.input().getType().cast<RankedTensorType>();
    inSize = inType.getShape()[1];
    auto weightType = op.weight().getType().cast<RankedTensorType>();
    inCh = weightType.getShape()[3];
    outCh = weightType.getShape()[0];
    auto outType = op.output().getType().cast<RankedTensorType>();
    outSize = outType.getShape()[1];
    kernelSize = weightType.getShape()[1];
    pad = op.pad()[0].dyn_cast<IntegerAttr>().getInt();
    stride = op.stride()[0].dyn_cast<IntegerAttr>().getInt();
    dilation = op.dilation()[0].dyn_cast<IntegerAttr>().getInt();
  }

  bool equalAttr(ConvHelper &rhs) {
    return (pad == rhs.pad) && (stride == rhs.stride) &&
           (dilation == rhs.dilation) && (kernelSize == rhs.kernelSize);
  }

  bool equalShape(ConvHelper &rhs) {
    return (inSize == rhs.inSize) && (outSize == rhs.outSize) &&
           (inCh == rhs.inCh) && (outCh == rhs.outCh) &&
           (kernelSize == rhs.kernelSize);
  }

  void takeSmallerDim(ConvHelper &rhs) {
    inCh = inCh < rhs.inCh ? inCh : rhs.inCh;
    outCh = outCh < rhs.outCh ? outCh : rhs.outCh;
    inSize = inSize < rhs.inSize ? inSize : rhs.inSize;
    outSize = inSize < rhs.inSize ? outSize : rhs.outSize;
  }

  bool operator==(ConvHelper &rhs) {
    if (this->isEmptyKey() || rhs.isEmptyKey()) {
      if (this->isEmptyKey() && rhs.isEmptyKey()) {
        return true;
      }
      return false;
    }
    if (this->isTombstoneKey() || rhs.isTombstoneKey()) {
      if (this->isTombstoneKey() && rhs.isTombstoneKey()) {
        return true;
      }
      return false;
    }
    return this->equalAttr(rhs);
  }

  unsigned getHashValue() const {
    unsigned hash = inCh * 37U;
    hash = (hash + inSize) * 37U;
    hash = (hash + outCh) * 37U;
    hash = (hash + outSize) * 37U;
    hash = (hash + kernelSize) * 37U;
    hash = (hash + pad) * 37U;
    hash = (hash + stride) * 37U;
    hash = (hash + dilation) * 37U;
    return hash;
  }

  bool operator<(const ConvHelper &rhs) const {
    ConvHelper lhs = ConvHelper(*this);
    ConvHelper rhsCopy = ConvHelper(rhs);
    if (lhs == rhsCopy) {
      return false;
    } else {
      return (this->getHashValue() < rhs.getHashValue());
    }
  }
  bool isEmptyKey() {
    int64_t emptyKey = (1UL << (sizeof(int64_t) * 8 - 1)) - 1UL;
    return (inCh == emptyKey) && (inSize == emptyKey) && (outCh == emptyKey) &&
           (outSize == emptyKey) && (kernelSize == emptyKey) &&
           (pad == emptyKey) && (stride == emptyKey) && (dilation == emptyKey);
  }
  bool isTombstoneKey() {
    int64_t tombstoneKey = (1UL << (sizeof(int64_t) * 8 - 1)) - 1UL - 1L;
    return (inCh == tombstoneKey) && (inSize == tombstoneKey) &&
           (outCh == tombstoneKey) && (outSize == tombstoneKey) &&
           (kernelSize == tombstoneKey) && (pad == tombstoneKey) &&
           (stride == tombstoneKey) && (dilation == tombstoneKey);
  }
};

namespace llvm {
template <> struct DenseMapInfo<ConvHelper> {
  static ConvHelper getEmptyKey() {
    int64_t emptyKey = (1UL << (sizeof(int64_t) * 8 - 1)) - 1UL;
    return ConvHelper{emptyKey, emptyKey, emptyKey, emptyKey,
                      emptyKey, emptyKey, emptyKey, emptyKey};
  }
  static ConvHelper getTombstoneKey() {
    int64_t tombstoneKey = (1UL << (sizeof(int64_t) * 8 - 1)) - 1UL - 1L;
    return ConvHelper{tombstoneKey, tombstoneKey, tombstoneKey, tombstoneKey,
                      tombstoneKey, tombstoneKey, tombstoneKey, tombstoneKey};
  }
  static unsigned getHashValue(ConvHelper Val) { return 0; }
  static bool isEqual(ConvHelper LHS, ConvHelper RHS) { return LHS == RHS; }
};
} // namespace llvm

static FuncOp createSharedFunction(ModuleOp module, ConvHelper sharedHelper,
                                   StringRef functionName) {
  auto builder = OpBuilder(module);

  // Create a shared function that contains sharedHelper's convolution
  SmallVector<Type, 16> inputTypes;
  auto inputShape = ArrayRef<int64_t>(
      {1, sharedHelper.inSize, sharedHelper.inSize, sharedHelper.inCh});
  auto inputType = RankedTensorType::get((inputShape), builder.getF32Type());
  inputTypes.push_back(inputType);
  auto weightShape =
      ArrayRef<int64_t>({sharedHelper.outCh, sharedHelper.kernelSize,
                         sharedHelper.kernelSize, sharedHelper.inCh});
  auto weightType = RankedTensorType::get((weightShape), builder.getF32Type());
  inputTypes.push_back(weightType);
  auto biasShape = ArrayRef<int64_t>({sharedHelper.outCh});
  auto biasType = RankedTensorType::get((biasShape), builder.getF32Type());
  inputTypes.push_back(biasType);

  auto resultShape = ArrayRef<int64_t>(
      {1, sharedHelper.outSize, sharedHelper.outSize, sharedHelper.outCh});
  auto resultType = RankedTensorType::get((resultShape), builder.getF32Type());

  auto newType = builder.getFunctionType(inputTypes, resultType);
  builder.setInsertionPointToStart(module.getBody());
  auto newFuncOp =
      builder.create<FuncOp>(builder.getUnknownLoc(), functionName, newType);
  newFuncOp->setAttr("shared", UnitAttr::get(newFuncOp->getContext()));
  newFuncOp->setAttr("name", builder.getStringAttr(functionName));
  newFuncOp->setAttr("count", builder.getI64IntegerAttr(0));
  auto entryBlock = newFuncOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Create Conv2DOp inside the created function/
  auto input = entryBlock->getArgument(0);
  auto weight = entryBlock->getArgument(1);
  auto bias = entryBlock->getArgument(2);
  auto outputType = newFuncOp.getResultTypes()[0];
  // auto pad = builder.getI64ArrayAttr({0, 0, 0, 0});
  auto pad = builder.getI64ArrayAttr(
      {sharedHelper.pad, sharedHelper.pad, sharedHelper.pad, sharedHelper.pad});
  auto stride =
      builder.getI64ArrayAttr({sharedHelper.stride, sharedHelper.stride});
  auto dilation =
      builder.getI64ArrayAttr({sharedHelper.dilation, sharedHelper.dilation});
  auto newConvOp =
      builder.create<tosa::Conv2DOp>(builder.getUnknownLoc(), outputType, input,
                                     weight, bias, pad, stride, dilation);

  // Create ReturnOp inside the created function/
  builder.create<func::ReturnOp>(builder.getUnknownLoc(), newConvOp.output());

  return newFuncOp;
}

static bool replaceFunction(ModuleOp module, ConvHelper sharedHelper,
                            FuncOp newFuncOp) {
  auto builder = OpBuilder(module);

  // Shared function name
  auto functionName = newFuncOp->getAttr("name").dyn_cast<StringAttr>();
  // Record ops to be erased.
  SmallVector<Operation *, 32> opToErase;

  // Convert matching convolutions into CallOp to shared function.
  for (auto func : module.getOps<FuncOp>()) {
    if (func->getAttr("shared"))
      continue;

    func.walk([&](tosa::Conv2DOp Conv2DOp) {
      auto loc = Conv2DOp.getLoc();
      ConvHelper currHelper = ConvHelper(Conv2DOp);

      if (!currHelper.equalAttr(sharedHelper))
        return;

      int64_t outChDiv =
          (currHelper.outCh + sharedHelper.outCh - 1) / sharedHelper.outCh;
      int64_t inChDiv =
          (currHelper.inCh + sharedHelper.inCh - 1) / sharedHelper.inCh;
      int64_t inSizeDiv =
          (currHelper.inSize + sharedHelper.inSize - 1) / sharedHelper.inSize;

      // Define zero bias if more than 1 input channel divs
      builder.setInsertionPoint(Conv2DOp);
      Value zeroBias;
      if (inChDiv > 1) {
        auto biasType =
            RankedTensorType::get(sharedHelper.outCh, builder.getF32Type());
        auto biasAttr = DenseFPElementsAttr::get(
            biasType, std::vector<float>(sharedHelper.outCh));
        zeroBias = builder.create<tosa::ConstOp>(loc, biasType, biasAttr);
      }

      // Output MemRef object allocated off-chip and original buffers
      bufferization::ToMemrefOp outMemref;
      for (auto user : Conv2DOp.output().getUsers()) {
        if (auto memref = dyn_cast<bufferization::ToMemrefOp>(user)) {
          outMemref = memref;
        }
      }
      memref::CopyOp outCopy;
      for (auto user : outMemref.memref().getUsers()) {
        if (auto copy = dyn_cast<memref::CopyOp>(user)) {
          outCopy = copy;
        }
      }
      bufferization::ToTensorOp outTensor;
      for (auto user : outCopy.target().getUsers()) {
        if (auto copy = dyn_cast<bufferization::ToTensorOp>(user)) {
          outTensor = copy;
        }
      }

      // Create output channel loop
      auto outChLoop = builder.create<AffineForOp>(loc, 0, outChDiv, 1);
      builder.setInsertionPointToStart(outChLoop.getBody());
      auto outCh =
          builder
              .create<AffineApplyOp>(
                  loc,
                  AffineMap::get(
                      1, 0, builder.getAffineDimExpr(0) * sharedHelper.outCh),
                  outChLoop.getInductionVar())
              .getODSResults(0)[0];

      // Create input channel loop
      auto inChLoop = builder.create<AffineForOp>(loc, 0, inChDiv, 1);
      builder.setInsertionPointToStart(inChLoop.getBody());
      auto inCh =
          builder
              .create<AffineApplyOp>(
                  loc,
                  AffineMap::get(
                      1, 0, builder.getAffineDimExpr(0) * sharedHelper.inCh),
                  inChLoop.getInductionVar())
              .getODSResults(0)[0];

      // Create width loop
      auto widthLoop = builder.create<AffineForOp>(loc, 0, inSizeDiv, 1);
      builder.setInsertionPointToStart(widthLoop.getBody());
      auto inWidth =
          builder
              .create<AffineApplyOp>(
                  loc,
                  AffineMap::get(
                      1, 0, builder.getAffineDimExpr(0) * sharedHelper.inSize),
                  widthLoop.getInductionVar())
              .getODSResults(0)[0];
      auto outWidth =
          builder
              .create<AffineApplyOp>(
                  loc,
                  AffineMap::get(
                      1, 0, builder.getAffineDimExpr(0) * sharedHelper.outSize),
                  widthLoop.getInductionVar())
              .getODSResults(0)[0];

      // Create height loop
      auto heightLoop = builder.create<AffineForOp>(loc, 0, inSizeDiv, 1);
      builder.setInsertionPointToStart(heightLoop.getBody());
      auto inHeight =
          builder
              .create<AffineApplyOp>(
                  loc,
                  AffineMap::get(
                      1, 0, builder.getAffineDimExpr(0) * sharedHelper.inSize),
                  heightLoop.getInductionVar())
              .getODSResults(0)[0];
      auto outHeight =
          builder
              .create<AffineApplyOp>(
                  loc,
                  AffineMap::get(
                      1, 0, builder.getAffineDimExpr(0) * sharedHelper.outSize),
                  heightLoop.getInductionVar())
              .getODSResults(0)[0];

      // Slice inputs
      auto bufOffset = ArrayRef<OpFoldResult>(
          {builder.getI64IntegerAttr(0), inWidth, inHeight, inCh});
      auto bufSize = ArrayRef<OpFoldResult>(
          {builder.getI64IntegerAttr(1),
           builder.getI64IntegerAttr(sharedHelper.inSize),
           builder.getI64IntegerAttr(sharedHelper.inSize),
           builder.getI64IntegerAttr(sharedHelper.inCh)});
      auto bufStride = ArrayRef<OpFoldResult>(
          {builder.getI64IntegerAttr(1), builder.getI64IntegerAttr(1),
           builder.getI64IntegerAttr(1), builder.getI64IntegerAttr(1)});
      auto slicedInput =
          builder
              .create<tensor::ExtractSliceOp>(loc, Conv2DOp.input(), bufOffset,
                                              bufSize, bufStride)
              .result();

      // Slice weights
      bufOffset = ArrayRef<OpFoldResult>({outCh, builder.getI64IntegerAttr(0),
                                          builder.getI64IntegerAttr(0), inCh});
      bufSize = ArrayRef<OpFoldResult>(
          {builder.getI64IntegerAttr(sharedHelper.outCh),
           builder.getI64IntegerAttr(sharedHelper.kernelSize),
           builder.getI64IntegerAttr(sharedHelper.kernelSize),
           builder.getI64IntegerAttr(sharedHelper.inCh)});
      bufStride = ArrayRef<OpFoldResult>(
          {builder.getI64IntegerAttr(1), builder.getI64IntegerAttr(1),
           builder.getI64IntegerAttr(1), builder.getI64IntegerAttr(1)});
      auto slicedWeight =
          builder
              .create<tensor::ExtractSliceOp>(loc, Conv2DOp.weight(), bufOffset,
                                              bufSize, bufStride)
              .result();

      // Slice biases
      bufOffset = ArrayRef<OpFoldResult>({outCh});
      bufSize = ArrayRef<OpFoldResult>(
          {builder.getI64IntegerAttr(sharedHelper.outCh)});
      bufStride = ArrayRef<OpFoldResult>({builder.getI64IntegerAttr(1)});
      auto slicedBias =
          builder
              .create<tensor::ExtractSliceOp>(loc, Conv2DOp.bias(), bufOffset,
                                              bufSize, bufStride)
              .result();

      // Call function
      auto operands = {slicedInput, slicedWeight, slicedBias};
      auto outType = RankedTensorType::get(
          {1, sharedHelper.outSize, sharedHelper.outSize, sharedHelper.outCh},
          builder.getF32Type());
      auto slicedOutTensor =
          builder.create<func::CallOp>(loc, functionName, outType, operands)
              .getODSResults(0)[0];
      auto count = newFuncOp->getAttr("count").dyn_cast<IntegerAttr>().getInt();
      newFuncOp->setAttr(
          "count", builder.getI64IntegerAttr(
                       count + outChDiv * inChDiv * inSizeDiv * inSizeDiv));

      // Bufferize result to memref
      auto slicedOutType =
          slicedOutTensor.getType().dyn_cast<RankedTensorType>();
      auto slicedOutMemref = builder.create<bufferization::ToMemrefOp>(
          loc,
          MemRefType::get(slicedOutType.getShape(),
                          slicedOutType.getElementType()),
          slicedOutTensor);

      // Create a subview of the final output memref
      bufOffset = ArrayRef<OpFoldResult>(
          {builder.getI64IntegerAttr(0), outWidth, outHeight, outCh});
      bufSize = ArrayRef<OpFoldResult>(
          {builder.getI64IntegerAttr(1),
           builder.getI64IntegerAttr(sharedHelper.outSize),
           builder.getI64IntegerAttr(sharedHelper.outSize),
           builder.getI64IntegerAttr(sharedHelper.outCh)});
      bufStride = ArrayRef<OpFoldResult>(
          {builder.getI64IntegerAttr(1), builder.getI64IntegerAttr(1),
           builder.getI64IntegerAttr(1), builder.getI64IntegerAttr(1)});
      auto subviewOut = builder.create<memref::SubViewOp>(
          loc, outCopy.target(), bufOffset, bufSize, bufStride);

      // Copy to sliced output
      builder.create<memref::CopyOp>(loc, slicedOutMemref, subviewOut);

      opToErase.push_back(outCopy);
      opToErase.push_back(outMemref);
      opToErase.push_back(Conv2DOp);
    });
  }

  // Erase all ops on the list.
  for (auto op : opToErase)
    op->erase();

  return true;
}

bool scalehls::applyShareTensorOperation(ModuleOp module, unsigned numTargets) {
  auto builder = OpBuilder(module);

  // Move all hidden features to off-chip buffer
  SmallVector<Operation *, 32> opToErase;
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](tosa::Conv2DOp Conv2DOp) {
      auto loc = Conv2DOp.getLoc();

      // Move results to off-chip
      auto outType = Conv2DOp.output().getType().dyn_cast<RankedTensorType>();
      auto outMemrefArg = func.front().addArgument(
          MemRefType::get(outType.getShape(), outType.getElementType()), loc);
      func.setType(builder.getFunctionType(
          func.front().getArgumentTypes(),
          func.back().getTerminator()->getOperandTypes()));

      builder.setInsertionPointAfter(Conv2DOp);
      auto newOp = builder.insert(Conv2DOp.clone());
      auto newConv2DOp = dyn_cast<tosa::Conv2DOp>(newOp);

      // Bufferize result to memref
      auto outMemref = builder.create<bufferization::ToMemrefOp>(
          loc, MemRefType::get(outType.getShape(), outType.getElementType()),
          newConv2DOp.output());

      // Copy to sliced output
      builder.create<memref::CopyOp>(loc, outMemref, outMemrefArg);

      // Create final output tensor
      auto outTensor =
          builder.create<bufferization::ToTensorOp>(loc, outType, outMemrefArg)
              .result();

      opToErase.push_back(Conv2DOp);
      Conv2DOp.replaceAllUsesWith(outTensor);
      return;
    });
  }
  // Erase all ops on the list.
  for (auto op : opToErase)
    op->erase();

  // Count the number of each shape of convolution.
  DenseMap<ConvHelper, std::pair<ConvHelper, unsigned>> countMap;

  // Traverse the entire module and count all the convolutions.
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](tosa::Conv2DOp Conv2DOp) {
      ConvHelper info = ConvHelper(Conv2DOp);
      if (!countMap.count(info)) {
        countMap[info] = std::pair<ConvHelper, unsigned>(info, 1);
      } else {
        info.takeSmallerDim(countMap[info].first);
        auto currCount = countMap[info].second;
        countMap.erase(info);
        countMap[info] = std::pair<ConvHelper, unsigned>(info, currCount + 1);
      }
    });
  }

  // Find the types of convolutions that happen frequently and replace it with
  // shared function
  ConvHelper sharedHelper;
  for (unsigned i = 0; i < numTargets; i++) {
    unsigned maxCount = 0;
    for (auto item : countMap) {
      if (item.second.second > maxCount) {
        maxCount = item.second.second;
        sharedHelper = item.first;
      }
    }
    if (maxCount != 0) {
      countMap.erase(sharedHelper);
      auto functionName = "shared_function_" + std::to_string(i);
      auto newFuncOp = createSharedFunction(module, sharedHelper, functionName);
      replaceFunction(module, sharedHelper, newFuncOp);
    }
  }

  return true;
}

namespace {
struct ShareTensorOperation
    : public ShareTensorOperationBase<ShareTensorOperation> {
  void runOnOperation() override {
    auto module = getOperation();
    applyShareTensorOperation(module, numTargets);
  }
};
} // namespace

std::unique_ptr<Pass> scalehls::createShareTensorOperationPass() {
  return std::make_unique<ShareTensorOperation>();
}
