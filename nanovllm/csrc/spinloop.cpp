// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// 基于 vLLM csrc/spinloop.cpp；使用当前解释器的 CPython API，无需 SABI 3.11。
#include <Python.h>

#include <time.h>

#if defined(__i386__) || defined(__x86_64__)
#include <cpuid.h>
#include <x86intrin.h>
#endif

namespace {

#if defined(CLOCK_MONOTONIC_RAW)
constexpr clockid_t kTimeoutClock = CLOCK_MONOTONIC_RAW;
#else
constexpr clockid_t kTimeoutClock = CLOCK_MONOTONIC;
#endif
constexpr unsigned int kMwaitxTimeoutCycles = 1000000;

/** @brief 每个 Python 模块实例独立保存的 CPU 等待指令能力。 */
struct SpinloopState {
  bool mSupportsMonitorx;
  unsigned int mMaxMonitorLineSize;
};

/**
 * @brief 检测上游使用的 AMD MONITORX/MWAITX 路径，其余 CPU 使用普通暂停指令。
 * @param state 模块持有的 host 状态；本函数写入全部字段，不访问 GPU。
 */
void determine_cpu_support(SpinloopState* state) {
  state->mSupportsMonitorx = false;
  state->mMaxMonitorLineSize = 0;
#if defined(__i386__) || defined(__x86_64__)
  unsigned int eax, ebx, ecx, edx;
  // 先检查 CPU 厂商与扩展叶范围，避免在不支持的 CPU 上执行 MWAITX。
  if (__get_cpuid(0, &eax, &ebx, &ecx, &edx) == 1 &&
      ebx == 0x68747541 && edx == 0x69746e65 && ecx == 0x444d4163 &&
      __get_cpuid(0x80000000, &eax, &ebx, &ecx, &edx) == 1 &&
      eax >= 0x80000001 &&
      __get_cpuid(0x80000001, &eax, &ebx, &ecx, &edx) == 1 &&
      (ecx & (1u << 29)) != 0) {
    state->mSupportsMonitorx = true;
    if (__get_cpuid(5, &eax, &ebx, &ecx, &edx) == 1) {
      state->mMaxMonitorLineSize = ebx & 0xff;
    }
  }
#endif
}

/**
 * @brief 等待 Python 条件满足或超时，不修改被监视的共享内存。
 * @param self 当前原生模块，用于取得独立的 CPU 能力状态。
 * @param args buffer、callback 与可选 timeout；buffer 的导出引用保持到返回。
 * @param kwargs 可选关键字参数，签名与 vLLM 的 spinloop 保持一致。
 * @return 条件满足返回 Py_True，超时返回 Py_False；异常返回 nullptr 并保留异常。
 */
PyObject* method_spinloop(PyObject* self, PyObject* args, PyObject* kwargs) {
  auto* state = static_cast<SpinloopState*>(PyModule_GetState(self));
  if (state == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to retrieve spinloop module state");
    return nullptr;
  }
  Py_buffer buffer;
  PyObject* callback;
  double timeout = 0.0;
  static const char* keywords[] = {"buffer", "callback", "timeout", nullptr};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "y*O|d",
                                   const_cast<char**>(keywords),
                                   &buffer, &callback, &timeout)) {
    return nullptr;
  }
  // 所有取得 buffer 后的退出路径都释放导出引用，避免阻止共享内存关闭。
  if (!PyCallable_Check(callback)) {
    PyErr_SetString(PyExc_TypeError, "callback parameter must be callable");
    PyBuffer_Release(&buffer);
    return nullptr;
  }
  timespec start;
  if (clock_gettime(kTimeoutClock, &start) != 0) {
    PyErr_SetFromErrno(PyExc_OSError);
    PyBuffer_Release(&buffer);
    return nullptr;
  }
  bool ready = false;
  bool error = false;
  const bool have_timeout = timeout > 1e-9;
  const bool buffer_qualifies =
      buffer.len > 0 && buffer.len <= state->mMaxMonitorLineSize;
  unsigned int iteration = 0;
  while (true) {
    // 回调访问 Python 对象时持有 GIL，暂停等待时才释放 GIL。
    PyObject* result = PyObject_CallNoArgs(callback);
    if (result == nullptr) {
      error = true;
      break;
    }
    const bool condition_met = result == Py_True;
    Py_DECREF(result);
    if (condition_met) {
      ready = true;
      break;
    }
    // 沿用上游节流方式：每 16 次检查时钟，避免每次轮询都读取系统时间。
    if (have_timeout && (iteration & 15u) == 0) {
      timespec now;
      if (clock_gettime(kTimeoutClock, &now) != 0) {
        PyErr_SetFromErrno(PyExc_OSError);
        error = true;
        break;
      }
      const double elapsed = static_cast<double>(now.tv_sec - start.tv_sec) +
                             static_cast<double>(now.tv_nsec - start.tv_nsec) * 1e-9;
      if (elapsed >= timeout) {
        break;
      }
    }
    ++iteration;
#if defined(__i386__) || defined(__x86_64__)
    if (buffer_qualifies && state->mSupportsMonitorx) {
      _mm_monitorx(buffer.buf, 0, 0);
      // 设置监视后重新检查，避免错过设置监视期间发生的写入。
      result = PyObject_CallNoArgs(callback);
      if (result == nullptr) {
        error = true;
        break;
      }
      const bool condition_after_monitor = result == Py_True;
      Py_DECREF(result);
      if (condition_after_monitor) {
        ready = true;
        break;
      }
      Py_BEGIN_ALLOW_THREADS
      _mm_mwaitx(1u << 1, 0, kMwaitxTimeoutCycles);
      Py_END_ALLOW_THREADS
    } else {
#endif
      // 其余平台保留上游忙等回退；释放 GIL，允许进程内其他 Python 线程推进。
      Py_BEGIN_ALLOW_THREADS
#if defined(__i386__) || defined(__x86_64__)
      __builtin_ia32_pause();
#elif defined(__aarch64__)
      __asm__ volatile("yield" ::: "memory");
#endif
      Py_END_ALLOW_THREADS
#if defined(__i386__) || defined(__x86_64__)
    }
#endif
  }
  PyBuffer_Release(&buffer);
  if (error) {
    return nullptr;
  }
  return PyBool_FromLong(ready);
}

PyMethodDef kSpinloopMethods[] = {
    {"spinloop", reinterpret_cast<PyCFunction>(method_spinloop),
     METH_VARARGS | METH_KEYWORDS, "Wait for a shared-memory condition or timeout."},
    {nullptr, nullptr, 0, nullptr}};

PyModuleDef kSpinloopModule = {
    PyModuleDef_HEAD_INIT, "nanovllm_spinloop", "CPU shared-memory waiting helpers.",
    sizeof(SpinloopState), kSpinloopMethods, nullptr, nullptr, nullptr, nullptr};

}  // namespace

/** @brief 初始化当前 Python 版本的扩展模块，并探测 host CPU 能力。 */
PyMODINIT_FUNC PyInit_nanovllm_spinloop() {
  PyObject* module = PyModule_Create(&kSpinloopModule);
  if (module != nullptr) {
    auto* state = static_cast<SpinloopState*>(PyModule_GetState(module));
    if (state != nullptr) {
      determine_cpu_support(state);
    }
  }
  return module;
}
