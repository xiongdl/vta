"""Old-style Relay external compiler registration for standalone VTA."""

import os

import numpy as np
import tvm
from tvm import relay
from tvm.contrib import cc, utils

from .conv2d import compile_qnn_conv2d
from .dense import _find_call, compile_qnn_dense
from ..relay import partition_for_vta


def _c_array(name, value, ctype):
    flat = np.asarray(value).reshape(-1)
    data = ",".join(str(int(x)) for x in flat)
    return f"static const {ctype} {name}[{flat.size}] = {{{data}}};"


def _compile_wrapper(artifact, source):
    temp = utils.tempdir()
    kernel = temp.relpath("kernel.o")
    wrapper = temp.relpath("wrapper.c")
    library = temp.relpath("vta_ext.dylib" if os.uname().sysname == "Darwin" else "vta_ext.so")
    artifacts = artifact if isinstance(artifact, (list, tuple)) else [artifact]
    kernels = []
    for index, item in enumerate(artifacts):
        path = temp.relpath(f"kernel{index}.o")
        item.module.save(path)
        kernels.append(path)
    with open(wrapper, "w", encoding="utf-8") as output:
        output.write(source)
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    tvm_root = os.environ.get("TVM_SOURCE_DIR", os.path.join(repo, "..", "tvm"))
    options = [f"-I{repo}/include", f"-I{tvm_root}/include", f"-I{tvm_root}/3rdparty/dlpack/include"]
    if os.uname().sysname == "Darwin":
        options += ["-Wl,-undefined,dynamic_lookup"]
    cc.create_shared(library, [wrapper] + kernels, options=options, cc="/usr/bin/clang")
    return tvm.runtime.load_module(library)


_C_HEADER = """
#include <stddef.h>
#include <stdint.h>
#include <pthread.h>
#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>
extern void* VTABufferAlloc(size_t);
extern void VTABufferFree(void*);
extern void VTABufferCopy(const void*, size_t, void*, size_t, size_t, int);
"""


def _cache_source(prefix, input_bytes, weight_bytes, bias_bytes, output_bytes):
    return f"""
static void* {prefix}_di=0; static void* {prefix}_dw=0;
static void* {prefix}_db=0; static void* {prefix}_dout=0;
static pthread_mutex_t {prefix}_lock=PTHREAD_MUTEX_INITIALIZER;
static int {prefix}_init(void) {{
  if ({prefix}_di) return 0;
  {prefix}_di=VTABufferAlloc({input_bytes}); {prefix}_dw=VTABufferAlloc({weight_bytes});
  {prefix}_db=VTABufferAlloc({bias_bytes}); {prefix}_dout=VTABufferAlloc({output_bytes});
  return (!{prefix}_di||!{prefix}_dw||!{prefix}_db||!{prefix}_dout) ? -1 : 0;
}}
__attribute__((destructor)) static void {prefix}_release(void) {{
  if({prefix}_di)VTABufferFree({prefix}_di); if({prefix}_dw)VTABufferFree({prefix}_dw);
  if({prefix}_db)VTABufferFree({prefix}_db); if({prefix}_dout)VTABufferFree({prefix}_dout);
  {prefix}_di={prefix}_dw={prefix}_db={prefix}_dout=0;
}}
"""


def _dense_wrapper(function, symbol):
    artifact = compile_qnn_dense(function, name=f"{symbol}_kernel")
    batch, in_features = artifact.input_shape
    _, out_features = artifact.output_shape
    weight = artifact.packed_weight
    bias = artifact.packed_bias
    env = __import__("vta").get_env()
    source = f'''
{_C_HEADER}
{_c_array("vta_weight", weight, "int8_t")}
{_c_array("vta_bias", bias, "int32_t")}
{_cache_source("cache", batch * in_features, weight.nbytes, bias.nbytes, batch * out_features)}
extern int {symbol}_kernel(TVMValue*, int*, int, TVMValue*, int*, void*);
TVM_DLL int {symbol}(TVMValue* args, int* tcodes, int nargs,
                     TVMValue* ret, int* ret_tcode, void* resource) {{
  if (nargs != 2) return -1;
  DLTensor* input = (DLTensor*)args[0].v_handle;
  DLTensor* output = (DLTensor*)args[1].v_handle;
  pthread_mutex_lock(&cache_lock);
  if (cache_init()) {{ pthread_mutex_unlock(&cache_lock); return -1; }}
  void* packed_input=cache_di; void* packed_weight=cache_dw;
  void* packed_bias=cache_db; void* packed_output=cache_dout;
  int8_t* logical_input = (int8_t*)input->data;
  int8_t* logical_output = (int8_t*)output->data;
  int8_t host_input[{batch * in_features}];
  int8_t host_output[{batch * out_features}];
  for (int b = 0; b < {batch}; ++b)
    for (int k = 0; k < {in_features}; ++k) {{
      int po = b / {env.BATCH}, bi = b % {env.BATCH};
      int ko = k / {env.BLOCK_IN}, ki = k % {env.BLOCK_IN};
      host_input[((po * {in_features // env.BLOCK_IN} + ko) * {env.BATCH} + bi) * {env.BLOCK_IN} + ki] =
          logical_input[b * {in_features} + k];
    }}
  VTABufferCopy(host_input, 0, packed_input, 0, {batch * in_features}, 1);
  static int constants_ready=0;
  if(!constants_ready) {{ VTABufferCopy(vta_weight,0,packed_weight,0,{weight.nbytes},1);
    VTABufferCopy(vta_bias,0,packed_bias,0,{bias.nbytes},1); constants_ready=1; }}
  int64_t ds[4] = {{{batch // env.BATCH}, {in_features // env.BLOCK_IN}, {env.BATCH}, {env.BLOCK_IN}}};
  int64_t ws[4] = {{{out_features // env.BLOCK_OUT}, {in_features // env.BLOCK_IN}, {env.BLOCK_OUT}, {env.BLOCK_IN}}};
  int64_t os[4] = {{{batch // env.BATCH}, {out_features // env.BLOCK_OUT}, {env.BATCH}, {env.BLOCK_OUT}}};
  DLDevice dev = {{kDLExtDev, 0}};
  DLTensor tensors[4] = {{
    {{packed_input, dev, 4, {{kDLInt,8,1}}, ds, 0, 0}},
    {{packed_weight, dev, 4, {{kDLInt,8,1}}, ws, 0, 0}},
    {{packed_bias, dev, 4, {{kDLInt,32,1}}, os, 0, 0}},
    {{packed_output, dev, 4, {{kDLInt,8,1}}, os, 0, 0}}
  }};
  TVMValue kargs[4]; int kt[4] = {{kTVMDLTensorHandle,kTVMDLTensorHandle,kTVMDLTensorHandle,kTVMDLTensorHandle}};
  for (int i=0;i<4;++i) kargs[i].v_handle=&tensors[i];
  int rc = {symbol}_kernel(kargs, kt, 4, ret, ret_tcode, resource);
  if (!rc) VTABufferCopy(packed_output, 0, host_output, 0, {batch * out_features}, 2);
  if (!rc) for (int b=0;b<{batch};++b) for (int c=0;c<{out_features};++c) {{
    int po=b/{env.BATCH}, bi=b%{env.BATCH}, co=c/{env.BLOCK_OUT}, ci=c%{env.BLOCK_OUT};
    logical_output[b*{out_features}+c]=host_output[((po*{out_features // env.BLOCK_OUT}+co)*{env.BATCH}+bi)*{env.BLOCK_OUT}+ci];
  }}
  pthread_mutex_unlock(&cache_lock);
  return rc;
}}
'''
    return _compile_wrapper(artifact, source)


def _conv2d_wrapper(function, symbol):
    artifact = compile_qnn_conv2d(function, name=f"{symbol}_kernel")
    env = __import__("vta").get_env()
    n, channels, height, width = artifact.input_shape
    _, out_channels, out_height, out_width = artifact.output_shape
    pn, pic, _, _, pb, pi = artifact.packed_input_shape
    pon, poc, _, _, pob, po = artifact.packed_output_shape
    physical_in = pic * pi
    physical_out = poc * po
    weight, bias = artifact.packed_weight, artifact.packed_bias
    input_bytes = n * physical_in * height * width
    output_bytes = n * physical_out * out_height * out_width
    source = f'''
{_C_HEADER}
{_c_array("vta_weight", weight, "int8_t")}
{_c_array("vta_bias", bias, "int32_t")}
{_cache_source("cache", input_bytes, weight.nbytes, bias.nbytes, output_bytes)}
extern int {symbol}_kernel(TVMValue*, int*, int, TVMValue*, int*, void*);
TVM_DLL int {symbol}(TVMValue* args, int* tcodes, int nargs,
                     TVMValue* ret, int* ret_tcode, void* resource) {{
  if (nargs != 2) return -1;
  DLTensor* input=(DLTensor*)args[0].v_handle; DLTensor* output=(DLTensor*)args[1].v_handle;
  pthread_mutex_lock(&cache_lock);
  if(cache_init()) {{ pthread_mutex_unlock(&cache_lock); return -1; }}
  void* di=cache_di; void* dw=cache_dw; void* db=cache_db; void* dout=cache_dout;
  int8_t hi[{input_bytes}]; int8_t ho[{output_bytes}];
  for (int i=0;i<{input_bytes};++i) hi[i]=0;
  int8_t* src=(int8_t*)input->data; int8_t* dst=(int8_t*)output->data;
  for(int b=0;b<{n};++b) for(int c=0;c<{channels};++c)
    for(int y=0;y<{height};++y) for(int x=0;x<{width};++x) {{
      int bo=b/{env.BATCH}, bi=b%{env.BATCH}, co=c/{env.BLOCK_IN}, ci=c%{env.BLOCK_IN};
      int p=(((((bo*{pic}+co)*{height}+y)*{width}+x)*{env.BATCH}+bi)*{env.BLOCK_IN}+ci);
      hi[p]=src[((b*{channels}+c)*{height}+y)*{width}+x];
    }}
  VTABufferCopy(hi,0,di,0,{input_bytes},1); static int constants_ready=0;
  if(!constants_ready) {{ VTABufferCopy(vta_weight,0,dw,0,{weight.nbytes},1);
    VTABufferCopy(vta_bias,0,db,0,{bias.nbytes},1); constants_ready=1; }}
  int64_t ds[6]={{{pn},{pic},{height},{width},{pb},{pi}}};
  int64_t ws[6]={{{','.join(str(int(x)) for x in weight.shape)}}};
  int64_t os[6]={{{pon},{poc},{out_height},{out_width},{pob},{po}}};
  DLDevice dev={{kDLExtDev,0}}; DLTensor ts[4]={{{{di,dev,6,{{kDLInt,8,1}},ds,0,0}},
    {{dw,dev,6,{{kDLInt,8,1}},ws,0,0}},{{db,dev,6,{{kDLInt,32,1}},os,0,0}},
    {{dout,dev,6,{{kDLInt,8,1}},os,0,0}}}};
  TVMValue ka[4]; int kt[4]={{kTVMDLTensorHandle,kTVMDLTensorHandle,kTVMDLTensorHandle,kTVMDLTensorHandle}};
  for(int i=0;i<4;++i) ka[i].v_handle=&ts[i];
  int rc={symbol}_kernel(ka,kt,4,ret,ret_tcode,resource);
  if(!rc) VTABufferCopy(dout,0,ho,0,{output_bytes},2);
  if(!rc) for(int b=0;b<{n};++b) for(int c=0;c<{out_channels};++c)
    for(int y=0;y<{out_height};++y) for(int x=0;x<{out_width};++x) {{
      int bo=b/{env.BATCH},bi=b%{env.BATCH},co=c/{env.BLOCK_OUT},ci=c%{env.BLOCK_OUT};
      int p=(((((bo*{poc}+co)*{out_height}+y)*{out_width}+x)*{env.BATCH}+bi)*{env.BLOCK_OUT}+ci);
      dst[((b*{out_channels}+c)*{out_height}+y)*{out_width}+x]=ho[p];
    }}
  pthread_mutex_unlock(&cache_lock); return rc;
}}
'''
    return _compile_wrapper(artifact, source)


def _composite_chain(expr, composite):
    calls = []

    def visit(node):
        if not isinstance(node, relay.Call):
            return
        for arg in node.args:
            visit(arg)
        if isinstance(node.op, relay.Function) and str(node.op.attrs.get("Composite")) == composite:
            calls.append(node.op)

    visit(expr)
    return calls


def _conv2d_chain_wrapper(function, symbol, functions):
    env = __import__("vta").get_env()
    artifacts = [
        compile_qnn_conv2d(item, name=f"{symbol}_kernel_{index}")
        for index, item in enumerate(functions)
    ]
    first, last = artifacts[0], artifacts[-1]
    if any(
        left.packed_output_shape != right.packed_input_shape
        for left, right in zip(artifacts, artifacts[1:])
    ):
        raise ValueError("Adjacent VTA conv2d regions require matching packed boundary shapes")
    n, channels, height, width = first.input_shape
    _, out_channels, out_height, out_width = last.output_shape
    pn, pic, _, _, pb, pi = first.packed_input_shape
    pon, poc, _, _, pob, po = last.packed_output_shape
    input_bytes = int(np.prod(first.packed_input_shape))
    output_bytes = int(np.prod(last.packed_output_shape))
    middle_bytes = max(int(np.prod(item.packed_output_shape)) for item in artifacts)
    constants = []
    declarations = []
    init = []
    tensor_calls = []
    for index, item in enumerate(artifacts):
        constants += [
            _c_array(f"w{index}", item.packed_weight, "int8_t"),
            _c_array(f"b{index}", item.packed_bias, "int32_t"),
        ]
        declarations.append(
            f"static void* dw{index}=0; static void* db{index}=0; "
            f"extern int {symbol}_kernel_{index}(TVMValue*,int*,int,TVMValue*,int*,void*);"
        )
        init.append(
            f"dw{index}=VTABufferAlloc({item.packed_weight.nbytes});"
            f"db{index}=VTABufferAlloc({item.packed_bias.nbytes});"
            f"VTABufferCopy(w{index},0,dw{index},0,{item.packed_weight.nbytes},1);"
            f"VTABufferCopy(b{index},0,db{index},0,{item.packed_bias.nbytes},1);"
        )
        shape = item.packed_output_shape
        ws = ",".join(str(int(x)) for x in item.packed_weight.shape)
        os = ",".join(str(int(x)) for x in shape)
        input_buffer = "di" if index == 0 else ("mid0" if index % 2 else "mid1")
        output_buffer = "dout" if index == len(artifacts) - 1 else ("mid0" if index % 2 == 0 else "mid1")
        ds = ",".join(str(int(x)) for x in item.packed_input_shape)
        tensor_calls.append(f"""
  int64_t ds{index}[6]={{{ds}}},ws{index}[6]={{{ws}}},os{index}[6]={{{os}}};
  DLTensor ts{index}[4]={{{{{input_buffer},dev,6,{{kDLInt,8,1}},ds{index},0,0}},
    {{dw{index},dev,6,{{kDLInt,8,1}},ws{index},0,0}},{{db{index},dev,6,{{kDLInt,32,1}},os{index},0,0}},
    {{{output_buffer},dev,6,{{kDLInt,8,1}},os{index},0,0}}}};
  for(int i=0;i<4;++i) ka[i].v_handle=&ts{index}[i];
  if(!rc) rc={symbol}_kernel_{index}(ka,kt,4,ret,ret_tcode,resource);
""")
    source = f'''
{_C_HEADER}
{''.join(constants)}
{''.join(declarations)}
static void *di=0,*dout=0,*mid0=0,*mid1=0; static int ready=0;
static int allocation_count=0;
static pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;
__attribute__((destructor)) static void release(void){{
  if(di)VTABufferFree(di);if(dout)VTABufferFree(dout);if(mid0)VTABufferFree(mid0);if(mid1)VTABufferFree(mid1);
  {''.join(f'if(dw{i})VTABufferFree(dw{i});if(db{i})VTABufferFree(db{i});' for i in range(len(artifacts)))}
}}
TVM_DLL int {symbol}_allocation_count(TVMValue* a,int* t,int n,TVMValue* r,int* rt,void* x){{
  r->v_int64=allocation_count;*rt=kTVMArgInt;return 0;}}
TVM_DLL int {symbol}(TVMValue* args,int* tcodes,int nargs,TVMValue* ret,int* ret_tcode,void* resource){{
  if(nargs!=2)return -1; pthread_mutex_lock(&lock);
  if(!ready){{di=VTABufferAlloc({input_bytes});dout=VTABufferAlloc({output_bytes});
    mid0=VTABufferAlloc({middle_bytes});mid1=VTABufferAlloc({middle_bytes});{''.join(init)}allocation_count++;ready=1;}}
  int8_t hi[{input_bytes}],ho[{output_bytes}];for(int i=0;i<{input_bytes};++i)hi[i]=0;
  int8_t* src=(int8_t*)((DLTensor*)args[0].v_handle)->data;
  for(int b=0;b<{n};++b)for(int c=0;c<{channels};++c)for(int y=0;y<{height};++y)for(int x=0;x<{width};++x){{
    int bo=b/{env.BATCH},bi=b%{env.BATCH},co=c/{env.BLOCK_IN},ci=c%{env.BLOCK_IN};
    hi[(((((bo*{pic}+co)*{height}+y)*{width}+x)*{env.BATCH}+bi)*{env.BLOCK_IN}+ci)]=src[((b*{channels}+c)*{height}+y)*{width}+x];}}
  VTABufferCopy(hi,0,di,0,{input_bytes},1);DLDevice dev={{kDLExtDev,0}};TVMValue ka[4];
  int kt[4]={{kTVMDLTensorHandle,kTVMDLTensorHandle,kTVMDLTensorHandle,kTVMDLTensorHandle}};int rc=0;
{''.join(tensor_calls)}
  if(!rc)VTABufferCopy(dout,0,ho,0,{output_bytes},2);int8_t* dst=(int8_t*)((DLTensor*)args[1].v_handle)->data;
  if(!rc)for(int b=0;b<{n};++b)for(int c=0;c<{out_channels};++c)for(int y=0;y<{out_height};++y)for(int x=0;x<{out_width};++x){{
    int bo=b/{env.BATCH},bi=b%{env.BATCH},co=c/{env.BLOCK_OUT},ci=c%{env.BLOCK_OUT};
    dst[((b*{out_channels}+c)*{out_height}+y)*{out_width}+x]=ho[(((((bo*{poc}+co)*{out_height}+y)*{out_width}+x)*{env.BATCH}+bi)*{env.BLOCK_OUT}+ci)];}}
  pthread_mutex_unlock(&lock);return rc;}}
'''
    return _compile_wrapper(artifacts, source)


def register_external_codegen():
    """Register ``relay.ext.vta`` without modifying or rebuilding TVM."""
    def compiler(function):
        symbol = str(function.attrs["global_symbol"])
        conv_chain = _composite_chain(function.body, "vta.qnn_conv2d")
        if len(conv_chain) > 1:
            return _conv2d_chain_wrapper(function, symbol, conv_chain)
        if _find_call(function.body, "qnn.dense") is not None:
            return _dense_wrapper(function, symbol)
        if _find_call(function.body, "qnn.conv2d") is not None:
            return _conv2d_wrapper(function, symbol)
        raise ValueError("relay.ext.vta region contains no supported core operator")

    tvm._ffi.register_func("relay.ext.vta", compiler, override=True)


def build_graph(mod, params=None, target="llvm", config=None):
    """Build a standard Relay executor factory with out-of-tree VTA regions."""
    register_external_codegen()
    partitioned = partition_for_vta(mod, params=params, config=config)
    with tvm.transform.PassContext(opt_level=3):
        return relay.build(partitioned, target=target)
