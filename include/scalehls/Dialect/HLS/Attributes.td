//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#ifndef SCALEHLS_DIALECT_HLS_ATTRIBUTES_TD
#define SCALEHLS_DIALECT_HLS_ATTRIBUTES_TD

def Resource : HLSAttr<"Resource"> {
  let summary = "Resource utilization information";
  let description = [{
    This attribute includes the utilization of several important on-chip
    resources, such as LUT, DSP, BRAM, and URAM.
  }];

  let hasCustomAssemblyFormat = 1;
  let mnemonic = "r";
  let parameters = (ins "int64_t":$lut, "int64_t":$dsp, "int64_t":$bram );
}

def Timing : HLSAttr<"Timing"> {
  let summary = "Timing information";
  let description = [{
    This attribute indicates the timing of operations, including scheduling
    begin and end cycles, latency, and interval.
  }];

  let hasCustomAssemblyFormat = 1;
  let mnemonic = "t";
  let parameters = (ins "int64_t":$begin, "int64_t":$end, "int64_t":$latency,
                        "int64_t":$interval);
}

def LoopInfo : HLSAttr<"LoopInfo"> {
  let summary = "Loop information";
  let description = [{
    This attribute includes loop information used by estimator and optimizer,
    including flattened trip count, iteration latency, and minimum initial
    interval (II).
  }];

  let hasCustomAssemblyFormat = 1;
  let mnemonic = "l";
  let parameters = (ins "int64_t":$flattenTripCount, "int64_t":$iterLatency,
                        "int64_t":$minII);
}

def LoopDirective : HLSAttr<"LoopDirective"> {
  let summary = "Aggregation of loop directives";
  let description = [{
    This attribute includes all loop directives, including loop pipelining,
    target initial interval (II), dataflow, and flattening.
  }];

  let hasCustomAssemblyFormat = 1;
  let mnemonic = "ld";
  let parameters = (ins "bool":$pipeline, "int64_t":$targetII, "bool":$dataflow,
                        "bool":$flatten);
}

def FuncDirective : HLSAttr<"FuncDirective"> {
  let summary = "Aggregation of function directives";
  let description = [{
    This attribute includes all function directives, including function
    pipelining, target interval, and dataflow.
  }];

  let hasCustomAssemblyFormat = 1;
  let mnemonic = "fd";
  let parameters = (ins "bool":$pipeline, "int64_t":$targetInterval,
                        "bool":$dataflow);
}

#endif // SCALEHLS_DIALECT_HLS_ATTRIBUTES_TD
