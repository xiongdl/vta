"""Old-style Relay external compiler registration for standalone VTA."""

import os

import numpy as np
import tvm
from tvm import relay
from tvm.contrib import cc, utils

from .conv2d import compile_qnn_conv2d
from .alu import compile_add, compile_packed_add
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
extern void VTARuntimeShutdown(void);
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


def _residual_conv_wrapper(function, symbol, functions):
    """Compile residual Add inside the final Conv2d kernel (no host bridge)."""
    env = __import__("vta").get_env()
    convs = [compile_qnn_conv2d(
        item, name=f"{symbol}_conv_{index}", residual=index == len(functions) - 1
    ) for index, item in enumerate(functions)]
    first, last = convs[0], convs[-1]
    if first.input_shape != last.output_shape or first.packed_input_shape != last.packed_output_shape:
        if len(functions) == 3:
            return _projection_residual_wrapper(function, symbol, functions)
        raise ValueError("Fused VTA residual requires identity or three-Conv projection shape")
    if any(
        left.packed_output_shape != right.packed_input_shape
        for left, right in zip(convs, convs[1:])
    ):
        raise ValueError("Adjacent residual Conv2d kernels require matching packed shapes")
    n, channels, height, width = first.input_shape
    pn, pc, _, _, pb, pi = first.packed_input_shape
    packed_bytes = int(np.prod(first.packed_input_shape))
    constants, declarations, init, calls = [], [], [], []
    for index, item in enumerate(convs):
        constants += [_c_array(f"rw{index}", item.packed_weight, "int8_t"),
                      _c_array(f"rb{index}", item.packed_bias, "int32_t")]
        declarations.append(
            f"static void *rwbuf{index}=0,*rbbuf{index}=0;"
            f"extern int {symbol}_conv_{index}(TVMValue*,int*,int,TVMValue*,int*,void*);"
        )
        init.append(
            f"rwbuf{index}=VTABufferAlloc({item.packed_weight.nbytes});"
            f"rbbuf{index}=VTABufferAlloc({item.packed_bias.nbytes});"
            f"VTABufferCopy(rw{index},0,rwbuf{index},0,{item.packed_weight.nbytes},1);"
            f"VTABufferCopy(rb{index},0,rbbuf{index},0,{item.packed_bias.nbytes},1);"
        )
        input_buffer = "skip" if index == 0 else ("mid0" if index % 2 else "mid1")
        output_buffer = "branch" if index == len(convs) - 1 else ("mid0" if index % 2 == 0 else "mid1")
        ds = ",".join(str(int(x)) for x in item.packed_input_shape)
        ws = ",".join(str(int(x)) for x in item.packed_weight.shape)
        os = ",".join(str(int(x)) for x in item.packed_output_shape)
        shortcut_tensor = (
            f",{{skip,dev,6,{{kDLInt,8,1}},ros{index},0,0}}" if index == len(convs) - 1 else ""
        )
        argument_count = 5 if index == len(convs) - 1 else 4
        calls.append(f"""
  int64_t rds{index}[6]={{{ds}}},rws{index}[6]={{{ws}}},ros{index}[6]={{{os}}};
  DLTensor rts{index}[{argument_count}]={{{{{input_buffer},dev,6,{{kDLInt,8,1}},rds{index},0,0}},
    {{rwbuf{index},dev,6,{{kDLInt,8,1}},rws{index},0,0}},
    {{rbbuf{index},dev,6,{{kDLInt,32,1}},ros{index},0,0}}{shortcut_tensor},
    {{{output_buffer},dev,6,{{kDLInt,8,1}},ros{index},0,0}}}};
  for(int i=0;i<{argument_count};++i)ka[i].v_handle=&rts{index}[i];
  if(!rc)rc={symbol}_conv_{index}(ka,kt5,{argument_count},ret,ret_tcode,resource);
  {"if(!rc)rc=TVMSynchronize(kDLExtDev,0,NULL);" if index != len(convs) - 1 else ""}
""")
    source = f'''
{_C_HEADER}
{''.join(constants)}
{''.join(declarations)}
static void *skip=0,*branch=0,*mid0=0,*mid1=0;static int ready=0;
static int host_bridge_count=0;
static pthread_mutex_t residual_lock=PTHREAD_MUTEX_INITIALIZER;
__attribute__((destructor))static void residual_release(void){{
 if(skip)VTABufferFree(skip);if(branch)VTABufferFree(branch);
 if(mid0)VTABufferFree(mid0);if(mid1)VTABufferFree(mid1);
 {''.join(f'if(rwbuf{i})VTABufferFree(rwbuf{i});if(rbbuf{i})VTABufferFree(rbbuf{i});' for i in range(len(convs)))}}}
TVM_DLL int {symbol}(TVMValue* args,int* tcodes,int nargs,TVMValue* ret,int* ret_tcode,void* resource){{
 if(nargs!=2)return -1;pthread_mutex_lock(&residual_lock);
 if(!ready){{skip=VTABufferAlloc({packed_bytes});branch=VTABufferAlloc({packed_bytes});
  mid0=VTABufferAlloc({packed_bytes});mid1=VTABufferAlloc({packed_bytes});
  {''.join(init)}ready=1;}}
 int8_t hi[{packed_bytes}],ho[{packed_bytes}];for(int i=0;i<{packed_bytes};++i)hi[i]=0;
 int8_t* src=(int8_t*)((DLTensor*)args[0].v_handle)->data;
 for(int b=0;b<{n};++b)for(int c=0;c<{channels};++c)for(int y=0;y<{height};++y)for(int x=0;x<{width};++x){{
  int bo=b/{env.BATCH},bi=b%{env.BATCH},co=c/{env.BLOCK_IN},ci=c%{env.BLOCK_IN};
  hi[(((((bo*{pc}+co)*{height}+y)*{width}+x)*{env.BATCH}+bi)*{env.BLOCK_IN}+ci)]=src[((b*{channels}+c)*{height}+y)*{width}+x];}}
 VTABufferCopy(hi,0,skip,0,{packed_bytes},1);
 DLDevice dev={{kDLExtDev,0}};TVMValue ka[5];
 int kt5[5]={{kTVMDLTensorHandle,kTVMDLTensorHandle,kTVMDLTensorHandle,kTVMDLTensorHandle,kTVMDLTensorHandle}};int rc=0;
{''.join(calls)}
 if(!rc)VTABufferCopy(branch,0,ho,0,{packed_bytes},2);int8_t* dst=(int8_t*)((DLTensor*)args[1].v_handle)->data;
 if(!rc)for(int b=0;b<{n};++b)for(int c=0;c<{channels};++c)for(int y=0;y<{height};++y)for(int x=0;x<{width};++x){{
  int bo=b/{env.BATCH},bi=b%{env.BATCH},co=c/{env.BLOCK_OUT},ci=c%{env.BLOCK_OUT};
  dst[((b*{channels}+c)*{height}+y)*{width}+x]=ho[(((((bo*{pc}+co)*{height}+y)*{width}+x)*{env.BATCH}+bi)*{env.BLOCK_OUT}+ci)];}}
 pthread_mutex_unlock(&residual_lock);return rc;}}
'''
    return _compile_wrapper(convs, source)


def _projection_residual_wrapper(function, symbol, functions):
    """Compile main1/main2/projection as one external module with packed boundaries."""
    env = __import__("vta").get_env()
    main1 = compile_qnn_conv2d(functions[0], name=f"{symbol}_main1")
    main2 = compile_qnn_conv2d(functions[1], name=f"{symbol}_main2", residual=True)
    projection = compile_qnn_conv2d(functions[2], name=f"{symbol}_projection")
    if main1.packed_output_shape != main2.packed_input_shape:
        raise ValueError("Projection residual main branch packed shapes do not match")
    if projection.packed_input_shape != main1.packed_input_shape:
        raise ValueError("Projection and main branch must consume the same packed input")
    if projection.packed_output_shape != main2.packed_output_shape:
        raise ValueError("Projection and main branch output packed shapes do not match")
    artifacts = [main1, main2, projection]
    n, channels, height, width = main1.input_shape
    _, out_channels, out_height, out_width = main2.output_shape
    pn, pic, _, _, pb, pi = main1.packed_input_shape
    pon, poc, _, _, pob, po = main2.packed_output_shape
    input_bytes = int(np.prod(main1.packed_input_shape))
    middle_bytes = int(np.prod(main1.packed_output_shape))
    output_bytes = int(np.prod(main2.packed_output_shape))
    constants = []
    declarations = []
    initialization = []
    releases = []
    for index, item in enumerate(artifacts):
        constants += [_c_array(f"pw{index}", item.packed_weight, "int8_t"),
                      _c_array(f"pbias{index}", item.packed_bias, "int32_t")]
        declarations.append(
            f"static void *pwbuf{index}=0,*pbbuf{index}=0;"
            f"extern int {symbol}_{('main1','main2','projection')[index]}"
            "(TVMValue*,int*,int,TVMValue*,int*,void*);"
        )
        initialization.append(
            f"pwbuf{index}=VTABufferAlloc({item.packed_weight.nbytes});"
            f"pbbuf{index}=VTABufferAlloc({item.packed_bias.nbytes});"
            f"VTABufferCopy(pw{index},0,pwbuf{index},0,{item.packed_weight.nbytes},1);"
            f"VTABufferCopy(pbias{index},0,pbbuf{index},0,{item.packed_bias.nbytes},1);"
        )
        releases.append(f"if(pwbuf{index})VTABufferFree(pwbuf{index});if(pbbuf{index})VTABufferFree(pbbuf{index});")

    def shape(item, packed_input=True):
        value = item.packed_input_shape if packed_input else item.packed_output_shape
        return ",".join(str(int(x)) for x in value)

    def weight_shape(item):
        return ",".join(str(int(x)) for x in item.packed_weight.shape)

    source = f'''
{_C_HEADER}
{''.join(constants)}
{''.join(declarations)}
static void *pin=0,*pmid=0,*pshort=0,*pout=0;static int projection_ready=0;
static pthread_mutex_t projection_lock=PTHREAD_MUTEX_INITIALIZER;
__attribute__((destructor))static void projection_release(void){{
 if(pin)VTABufferFree(pin);if(pmid)VTABufferFree(pmid);if(pshort)VTABufferFree(pshort);if(pout)VTABufferFree(pout);
 {''.join(releases)}}}
TVM_DLL int {symbol}(TVMValue* args,int* tcodes,int nargs,TVMValue* ret,int* ret_tcode,void* resource){{
 if(nargs!=2)return -1;pthread_mutex_lock(&projection_lock);
 if(!projection_ready){{pin=VTABufferAlloc({input_bytes});pmid=VTABufferAlloc({middle_bytes});
  pshort=VTABufferAlloc({output_bytes});pout=VTABufferAlloc({output_bytes});
  {''.join(initialization)}projection_ready=1;}}
 int8_t hi[{input_bytes}],ho[{output_bytes}];for(int i=0;i<{input_bytes};++i)hi[i]=0;
 int8_t* src=(int8_t*)((DLTensor*)args[0].v_handle)->data;
 for(int b=0;b<{n};++b)for(int c=0;c<{channels};++c)for(int y=0;y<{height};++y)for(int x=0;x<{width};++x){{
  int bo=b/{env.BATCH},bi=b%{env.BATCH},co=c/{env.BLOCK_IN},ci=c%{env.BLOCK_IN};
  hi[(((((bo*{pic}+co)*{height}+y)*{width}+x)*{env.BATCH}+bi)*{env.BLOCK_IN}+ci)]=src[((b*{channels}+c)*{height}+y)*{width}+x];}}
 VTABufferCopy(hi,0,pin,0,{input_bytes},1);DLDevice dev={{kDLExtDev,0}};TVMValue ka[5];
 int kt[5]={{kTVMDLTensorHandle,kTVMDLTensorHandle,kTVMDLTensorHandle,kTVMDLTensorHandle,kTVMDLTensorHandle}};int rc=0;
 int64_t pds[6]={{{shape(projection)}}},pws[6]={{{weight_shape(projection)}}},pos[6]={{{shape(projection,False)}}};
 DLTensor pts[4]={{{{pin,dev,6,{{kDLInt,8,1}},pds,0,0}},{{pwbuf2,dev,6,{{kDLInt,8,1}},pws,0,0}},
  {{pbbuf2,dev,6,{{kDLInt,32,1}},pos,0,0}},{{pshort,dev,6,{{kDLInt,8,1}},pos,0,0}}}};
 for(int i=0;i<4;++i)ka[i].v_handle=&pts[i];if(!rc)rc={symbol}_projection(ka,kt,4,ret,ret_tcode,resource);
 if(!rc)rc=TVMSynchronize(kDLExtDev,0,NULL);
 int64_t d1[6]={{{shape(main1)}}},w1[6]={{{weight_shape(main1)}}},o1[6]={{{shape(main1,False)}}};
 DLTensor t1[4]={{{{pin,dev,6,{{kDLInt,8,1}},d1,0,0}},{{pwbuf0,dev,6,{{kDLInt,8,1}},w1,0,0}},
  {{pbbuf0,dev,6,{{kDLInt,32,1}},o1,0,0}},{{pmid,dev,6,{{kDLInt,8,1}},o1,0,0}}}};
 for(int i=0;i<4;++i)ka[i].v_handle=&t1[i];if(!rc)rc={symbol}_main1(ka,kt,4,ret,ret_tcode,resource);
 if(!rc)rc=TVMSynchronize(kDLExtDev,0,NULL);
 int64_t d2[6]={{{shape(main2)}}},w2[6]={{{weight_shape(main2)}}},o2[6]={{{shape(main2,False)}}};
 DLTensor t2[5]={{{{pmid,dev,6,{{kDLInt,8,1}},d2,0,0}},{{pwbuf1,dev,6,{{kDLInt,8,1}},w2,0,0}},
  {{pbbuf1,dev,6,{{kDLInt,32,1}},o2,0,0}},{{pshort,dev,6,{{kDLInt,8,1}},o2,0,0}},
  {{pout,dev,6,{{kDLInt,8,1}},o2,0,0}}}};
 for(int i=0;i<5;++i)ka[i].v_handle=&t2[i];if(!rc)rc={symbol}_main2(ka,kt,5,ret,ret_tcode,resource);
 if(!rc)VTABufferCopy(pout,0,ho,0,{output_bytes},2);int8_t* dst=(int8_t*)((DLTensor*)args[1].v_handle)->data;
 if(!rc)for(int b=0;b<{n};++b)for(int c=0;c<{out_channels};++c)for(int y=0;y<{out_height};++y)for(int x=0;x<{out_width};++x){{
  int bo=b/{env.BATCH},bi=b%{env.BATCH},co=c/{env.BLOCK_OUT},ci=c%{env.BLOCK_OUT};
  dst[((b*{out_channels}+c)*{out_height}+y)*{out_width}+x]=ho[(((((bo*{poc}+co)*{out_height}+y)*{out_width}+x)*{env.BATCH}+bi)*{env.BLOCK_OUT}+ci)];}}
 pthread_mutex_unlock(&projection_lock);return rc;}}
'''
    return _compile_wrapper(artifacts, source)


def _add_wrapper(function, symbol):
    artifact = compile_add(function, f"{symbol}_kernel")
    elements = int(np.prod(artifact.logical_shape)); padded = int(np.prod(artifact.packed_shape))
    source = f'''
{_C_HEADER}
extern int {symbol}_kernel(TVMValue*,int*,int,TVMValue*,int*,void*);
static void *a=0,*b=0,*o=0;static pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;
__attribute__((destructor)) static void release(void){{
 if(a)VTABufferFree(a);if(b)VTABufferFree(b);if(o)VTABufferFree(o);a=b=o=0;}}
TVM_DLL int {symbol}(TVMValue* args,int* tc,int n,TVMValue* r,int* rt,void* x){{
 if(n!=3)return -1;pthread_mutex_lock(&lock);if(!a){{a=VTABufferAlloc({padded*4});b=VTABufferAlloc({padded*4});o=VTABufferAlloc({padded});}}
 int32_t ha[{padded}],hb[{padded}];int8_t ho[{padded}];for(int i=0;i<{padded};++i)ha[i]=hb[i]=0;
 int8_t* ia=(int8_t*)((DLTensor*)args[0].v_handle)->data;int8_t* ib=(int8_t*)((DLTensor*)args[1].v_handle)->data;
 for(int i=0;i<{elements};++i){{ha[i]=ia[i];hb[i]=ib[i];}}VTABufferCopy(ha,0,a,0,{padded*4},1);VTABufferCopy(hb,0,b,0,{padded*4},1);
 int64_t s[4]={{{','.join(str(x) for x in artifact.packed_shape)}}};DLDevice d={{kDLExtDev,0}};
 DLTensor ts[3]={{{{a,d,4,{{kDLInt,32,1}},s,0,0}},{{b,d,4,{{kDLInt,32,1}},s,0,0}},{{o,d,4,{{kDLInt,8,1}},s,0,0}}}};
 TVMValue ka[3];int kt[3]={{kTVMDLTensorHandle,kTVMDLTensorHandle,kTVMDLTensorHandle}};for(int i=0;i<3;++i)ka[i].v_handle=&ts[i];
 int rc={symbol}_kernel(ka,kt,3,r,rt,x);if(!rc)VTABufferCopy(o,0,ho,0,{padded},2);int8_t* out=(int8_t*)((DLTensor*)args[2].v_handle)->data;
 if(!rc)for(int i=0;i<{elements};++i)out[i]=ho[i];pthread_mutex_unlock(&lock);return rc;}}
'''
    return _compile_wrapper(artifact, source)


def register_external_codegen():
    """Register ``relay.ext.vta`` without modifying or rebuilding TVM."""
    def compiler(function):
        symbol = str(function.attrs["global_symbol"])
        conv_chain = _composite_chain(function.body, "vta.qnn_conv2d")
        if len(conv_chain) > 1 and _find_call(function.body, "add") is not None:
            return _residual_conv_wrapper(function, symbol, conv_chain)
        if len(conv_chain) > 1:
            return _conv2d_chain_wrapper(function, symbol, conv_chain)
        if _find_call(function.body, "qnn.dense") is not None:
            return _dense_wrapper(function, symbol)
        if _find_call(function.body, "qnn.conv2d") is not None:
            return _conv2d_wrapper(function, symbol)
        if _find_call(function.body, "add") is not None:
            return _add_wrapper(function, symbol)
        raise ValueError("relay.ext.vta region contains no supported core operator")

    tvm._ffi.register_func("relay.ext.vta", compiler, override=True)


def build_graph(mod, params=None, target="llvm", config=None):
    """Build a standard Relay executor factory with out-of-tree VTA regions."""
    # Register the selected out-of-tree runtime before loading native wrappers
    # that resolve VTABuffer* symbols from it.
    from ..testing import simulator  # pylint: disable=import-outside-toplevel,unused-import

    register_external_codegen()
    partitioned = partition_for_vta(mod, params=params, config=config)
    with tvm.transform.PassContext(opt_level=3):
        return relay.build(partitioned, target=target)
