# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import paddle
from paddle.fluid.framework import _dygraph_tracer
try:
    import paddle.profiler as profiler
except:
    print("import paddle.profiler failed.")


GLOBAL_PROFILER = None
GLOBAL_PROFILER_TYPE = "none"
GLOBAL_PROFILER_STEP = -1
GLOBAL_PROFILER_START_STEP = -1
GLOBAL_PROFILER_END_STEP = -1
GLOBAL_EVENT_STACK = []


def set_profiler_type(prof, nvprof):
    global GLOBAL_PROFILER_TYPE
    if prof:
        GLOBAL_PROFILER_TYPE = "native"
    elif nvprof:
        GLOBAL_PROFILER_TYPE = "nvprof"
    #    GLOBAL_PROFILER_TYPE = "debug"


def memory_allocate_info():
    def to_giga_bytes(v):
        return float(v) / 1000000000.0

    max_allocated = to_giga_bytes(paddle.device.cuda.max_memory_allocated(device="gpu:0"))
    max_reserved = to_giga_bytes(paddle.device.cuda.max_memory_reserved(device="gpu:0"))
    allocated = to_giga_bytes(paddle.device.cuda.memory_allocated(device="gpu:0"))
    reserved = to_giga_bytes(paddle.device.cuda.memory_reserved(device="gpu:0"))
    memory_str = "[MEMORY] max_allocated={:.2f} GB, max_reserved={:.2f} GB, allocated={:.2f} GB, reserved={:.2f} GB".format(max_allocated, max_reserved, allocated, reserved)
    return memory_str


def dtype_info():
    default_dtype = paddle.get_default_dtype()
    amp_dtype = _dygraph_tracer()._amp_dtype
    dtype_info_str = "default_dtype={}, amp_dtype={}".format(default_dtype, amp_dtype)
    return dtype_info_str


def push_profile_event(name):
    #global GLOBAL_PROFILER_TYPE
    ##if GLOBAL_PROFILER_TYPE == "native":
    ##    record_event = profiler.RecordEvent(name)
    ##    record_event.begin()
    ##    return record_event
    ##elif GLOBAL_PROFILER_TYPE == "nvprof":
    ##    paddle.fluid.core.nvprof_nvtx_push(name)
    #if GLOBAL_PROFILER_TYPE == "debug":
    #    global GLOBAL_EVENT_STACK
    #    GLOBAL_EVENT_STACK.append(name)
    #    print("++++++++++++++++++++ Enter class {}: {}".format(name, dtype_info()), flush=True)
    #    sys.stdout.flush()
    return None


def pop_profile_event(event=None):
    pass
    #global GLOBAL_PROFILER_TYPE
    ##if GLOBAL_PROFILER_TYPE == "native":
    ##    event.end()
    ##elif GLOBAL_PROFILER_TYPE == "nvprof":
    ##    paddle.fluid.core.nvprof_nvtx_pop()
    #if GLOBAL_PROFILER_TYPE == "debug":
    #    global GLOBAL_EVENT_STACK
    #    name = GLOBAL_EVENT_STACK.pop()
    #    print("-------------------- Leave class {}: {}".format(name, dtype_info()), flush=True)
    #    sys.stdout.flush()
          

def _switch_profile(start, end, event_name=None):
    global GLOBAL_PROFILER_TYPE
    global GLOBAL_PROFILER_STEP
    GLOBAL_PROFILER_STEP += 1

    if GLOBAL_PROFILER_TYPE == "native-old":
        if GLOBAL_PROFILER_STEP == start:
            paddle.utils.profiler.start_profiler("All", "Default")
        elif GLOBAL_PROFILER_STEP == end:
            paddle.utils.profiler.stop_profiler("total", "alphafold.profile")
    elif GLOBAL_PROFILER_TYPE == "nvprof":
        if event_name is None:
            event_name = str(GLOBAL_PROFILER_STEP)
        if GLOBAL_PROFILER_STEP == start:
            paddle.fluid.core.nvprof_start()
            paddle.fluid.core.nvprof_enable_record_event()
            paddle.fluid.core.nvprof_nvtx_push(event_name)
        elif GLOBAL_PROFILER_STEP == end:
            paddle.fluid.core.nvprof_nvtx_pop()
            paddle.fluid.core.nvprof_stop()
        elif GLOBAL_PROFILER_STEP > start and GLOBAL_PROFILER_STEP < end:
            paddle.fluid.core.nvprof_nvtx_pop()
            paddle.fluid.core.nvprof_nvtx_push(event_name)


def get_profiler(start, end, profiler_type=None):
    global GLOBAL_PROFILER
    global GLOBAL_PROFILER_TYPE
    global GLOBAL_PROFILER_START_STEP
    global GLOBAL_PROFILER_END_STEP

    if profiler_type is not None:
        GLOBAL_PROFILER_TYPE = profiler_type
    else:
        GLOBAL_PROFILER_TYPE = "none"
    GLOBAL_PROFILER_START_STEP = start
    GLOBAL_PROFILER_END_STEP = end
    if profiler_type is not None and profiler_type == "native":
        timer_only = False
    else:
        timer_only = True
    if profiler_type is not None and profiler_type == "nvprof":
        enable_nvtx = True
    else:
        enable_nvtx = False
    try:
        print("[PROFILER] profiler_type={}, timer_only={}, enable_nvtx={}".format(profiler_type, timer_only, enable_nvtx))
        GLOBAL_PROFILER = profiler.Profiler(
            targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],                                                  
            scheduler=[start, end],
            timer_only=timer_only,
            emit_nvtx=enable_nvtx)
        GLOBAL_PROFILER.start()
    except:
        print("Error to define profiler!!!")
        pass


def step(event_name=None):
    global GLOBAL_PROFILER
    global GLOBAL_PROFILER_START_STEP
    global GLOBAL_PROFILER_END_STEP

    try:
        GLOBAL_PROFILER.step()
    except:
        pass

    print("[PROFILE] GLOBAL_PROFILER_STEP = {}".format(GLOBAL_PROFILER_STEP))
    _switch_profile(GLOBAL_PROFILER_START_STEP, GLOBAL_PROFILER_END_STEP, event_name)


def stop(op_detail=True):
    global GLOBAL_PROFILER
    try:
        GLOBAL_PROFILER.stop()
        GLOBAL_PROFILER.summary(op_detail=op_detail)
    except:
        pass
